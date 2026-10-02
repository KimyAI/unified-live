# Seed-VC realtime — bridge expérimental local

Ce bridge utilise uniquement `load_models` et `custom_infer` du fichier
[`real-time-gui.py` à la révision `51383ef`](https://github.com/Plachtaa/seed-vc/blob/51383efd921027683c89e5348211d93ff12ac2a8/real-time-gui.py).
Il n'instancie ni GUI, ni `sounddevice.Stream`, ni FunASR VAD. Le cœur Unified
Live garde la propriété du microphone et de la sortie audio. Aucune conversion
CUDA ni qualité perceptuelle n'a été validée sur le matériel cible.

## Installation locale, séparée du cœur

1. Placer un checkout Seed-VC à la révision ci-dessus dans un dossier distinct.
   Sa racine doit contenir `real-time-gui.py` et `configs/hifigan.yml`. Vérifier
   la licence de ce code et des poids avant usage.
2. Créer un environnement Python dédié compatible avec les exigences de cette
   révision et y installer ses dépendances ainsi qu'une version CUDA de PyTorch
   adaptée au pilote local. Ne pas installer ces dépendances dans l'environnement
   du cœur. Choisir les versions depuis la documentation du checkout et de
   PyTorch pour la machine concernée ; ce bridge n'en impose pas une combinaison
   CUDA prétendument validée.
3. Préparer localement les fichiers suivants, avec leur provenance, licence,
   révision et empreinte SHA-256 consignées hors des options applicatives :

   - `DiT_uvit_tat_xlsr_ema.pth` et le YAML realtime
     `config_dit_mel_seed_uvit_xlsr_tiny.yml` (22 050 Hz, hop 256) ;
   - le répertoire complet `facebook/wav2vec2-xls-r-300m`, dont `config.json`,
     `preprocessor_config.json` et les poids, éventuellement en shards ;
   - `campplus_cn_common.bin` (`funasr/campplus`) ;
   - `hift.pt` (`FunAudioLLM/CosyVoice-300M`) ;
   - un fichier de référence audio local d'au moins 0,5 s.

   Le bridge vérifie la présence de ces fichiers avant tout import Torch. Il
   active `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` et
   `HF_DATASETS_OFFLINE=1` avant les imports, fournit les deux checkpoints
   auxiliaires par un hook local, et remplace dans une copie temporaire du YAML
   le nom du modèle XLS-R par son chemin local. Un autre appel implicite au Hub
   est refusé. Il ne télécharge aucun poids.

4. Configurer `settings.backends["seed-vc"]` avec le Python de cet environnement,
   `root` égal au checkout et `module` égal à
   `unified_live.engines.voice.seed_vc.bridge`. Le worker lance ce module dans
   son propre processus avec le checkout comme répertoire courant. Exemple
   conceptuel, à adapter aux chemins réellement vérifiés :

   ```json
   {
     "python": "/chemin/seed-venv/bin/python",
     "root": "/chemin/seed-vc",
     "module": "unified_live.engines.voice.seed_vc.bridge",
     "startup_timeout": 120,
     "processing_timeout": 2
   }
   ```

   Le délai de traitement doit être réévalué d'après les p95 mesurés sur le GPU
   réel. Une valeur trop basse fait échouer le backend et déclenche le bypass
   visible. Les options du modèle sont transmises par `voice_options` :

   ```json
   {
     "seed_root": "/chemin/seed-vc",
     "checkpoint_path": "/chemin/models/DiT_uvit_tat_xlsr_ema.pth",
     "config_path": "/chemin/seed-vc/configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml",
     "reference_audio": "/chemin/reference.wav",
     "campplus_checkpoint": "/chemin/models/campplus_cn_common.bin",
     "hift_checkpoint": "/chemin/models/hift.pt",
     "xlsr_model_dir": "/chemin/models/wav2vec2-xls-r-300m",
     "sample_rate": 22050,
     "chunk_size": 3969,
     "gpu": 0,
     "fp16": true,
     "block_time": 0.18,
     "crossfade": 0.04,
     "content_context_left": 2.5,
     "context_left": 0.5,
     "context_right": 0.02,
     "diffusion_steps": 10,
     "cfg": 0.7,
     "max_prompt_length": 3.0
   }
   ```

   Sous Windows, les chemins utilisent la syntaxe Windows habituelle et
   l'exécutable est `Scripts\\python.exe`. Sélectionner Seed-VC exige que les
   réglages globaux audio soient exactement `sample_rate=22050` et
   `chunk_size=3969` pour `block_time=0.18`. Un autre `block_time` multiple de
   20 ms impose `chunk_size=round(block_time*50)*441`. Le bridge refuse une
   taille ou fréquence différente avant de charger les modèles.

## Temporalité et paramètres

Chaque appel reçoit un bloc mono `float32` et renvoie un bloc de même taille.
Le worker conserve la fenêtre 22,05 kHz, son historique rééchantillonné à
16 kHz, le cache de référence du modèle et le recouvrement SOLA. Il prépare les
buffers pendant environ 15 blocs (2,7 s avec l'exemple) et émet du silence
pendant ce remplissage. `is_warming_up` permet de l'indiquer explicitement ;
ce silence initial ne représente pas une latence permanente. Le premier appel
CUDA et le calcul de la référence sont exécutés au démarrage du worker sous son
délai d'initialisation, avant les blocs en direct.

Le changement ON/OFF de l'effet peut appeler `reset()` : il efface le contexte
micro et SOLA et recommence le warmup, tout en gardant modèles et référence en
mémoire. Changer la référence ou le checkpoint exige une nouvelle sélection du
moteur avec des options actualisées, donc un nouveau worker ; `load_source` et
`load_model` lèvent une erreur explicite au lieu de laisser un cache périmé.

`content_context_left` est le contexte de l'encodeur de contenu ;
`context_left` celui du DiT. Leur différence va à `custom_infer` comme
`cd_difference`. `context_right` est le tail exclu par la fonction upstream.
Le bridge réutilise exactement ses nombres `skip_head`, `skip_tail` et
`return_length` en trames de 20 ms. Il ne retire pas une seconde fois le tail.
Le raccord SOLA cherche le meilleur décalage dans 20 ms et applique le fondu
sinus carré de l'upstream.

Après un bloc, `get_algorithmic_delay_ms()` renvoie le décalage **nominal du
découpage** `(tail_droit + recouvrement + recherche - offset_SOLA) / 22050`.
Avec les valeurs ci-dessus, il varie de 60 à 80 ms. La capture d'un bloc de
180 ms et la durée d'inférence sont déjà mesurées par le pipeline hôte ; elles
ne doivent pas être additionnées deux fois. Le modèle peut déplacer des
phonèmes au-delà de ce calcul : seule une mesure clap/loopback sur les vrais
périphériques permet d'établir le décalage audiovisuel physique.

Ce profil `.18/.04/2.5/.5/.02` est expérimental pour Unified Live. Le preset
initial de la GUI upstream emploie notamment `block_time=0.3` et ne constitue
pas une preuve de qualité pour ce profil. Aucun VAD ou transcript n'est
utilisé ; les blocs silencieux passent aussi par le modèle après le warmup.

Sources de comportement et d'assets :
[realtime upstream](https://github.com/Plachtaa/seed-vc/blob/51383efd921027683c89e5348211d93ff12ac2a8/real-time-gui.py),
[YAML XLS-R realtime](https://github.com/Plachtaa/seed-vc/blob/51383efd921027683c89e5348211d93ff12ac2a8/configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml),
[note d'intégration](BACKEND_INTEGRATION.md).
