# Contrats d’intégration des backends upstream

Cette note prépare l’adaptation, sans créer les bridges. Les conclusions de code sont prises sur les révisions exactes suivantes : ReSwapper `44474cb8d85d15274ae274e5e0f5f44f0814af51` (14 juin 2025) et Seed-VC `51383efd921027683c89e5348211d93ff12ac2a8` (20 avril 2025). Les copies sources sans poids sont conservées temporairement dans `/tmp/unified-upstream/` avec un manifeste de hash; voir `manifests/upstream-source-manifest.json` dans ce dossier.

Les wrappers ci-dessous sont des contrats de conception. Les modèles ne sont ni inclus ni téléchargés par cette note. Les fichiers de poids doivent être vérifiés, licenciés, installés localement et leur empreinte figée avant d’activer les workers.

## Worker vision ReSwapper

### Entrées et sorties

Le contrat conseillé pour un worker séquentiel est :

- `configure(source_image, model_path, emap_path, resolution=128, device_id=0)` charge les dépendances une seule fois, détecte le visage source, calcule et garde son latent, charge `StyleTransferModel` et vérifie les formes/résolution.
- `process(frame_bgr_uint8)` reçoit une image BGR contiguë `H×W×3`, détecte une cible, renvoie la frame originale si aucune cible n’est détectée, sinon renvoie la frame composée avec le visage remplacé.
- `close()` libère explicitement le modèle et la session de détection si le worker doit être redémarré.

Dans les fichiers upstream `swap.py`, `Image.py`, `face_align.py`, `StyleTransferModel_128.py` et `iresnet.py`, les primitives réellement disponibles sont `load_model`, `create_source`, `create_target`, `swap_face`, `face_align.norm_crop2`, `Image.getBlob`, `Image.getLatent`, `Image.postprocess_face` et `Image.blend_swapped_image`. `create_target` ne gère pas l’absence de visage et `create_source` accepte un chemin plutôt qu’une frame déjà chargée ; l’adaptateur du worker doit appliquer des contrôles avant d’appeler ces fonctions.

Ordre d’un frame : `FaceAnalysis.get(frame)` → sélectionner explicitement une cible → `norm_crop2(frame, face.kps, resolution)` → `Image.getBlob(crop, (resolution,resolution))` → tenseurs Torch sur le device → `model(target, source_latent)` sans gradients → conversion BGR via `postprocess_face` → `blend_swapped_image(swapped, frame, M)`. L’embedding source est `face.normed_embedding.reshape(1,-1) @ emap`, normalisé L2 ; forme `[1,512]`. Le modèle accepte un crop cible `[1,3,R,R]` dans `[0,1]` et ce latent. En résolution 128, le chemin est le mieux documenté. Le README signale un défaut d’alignement pour d’autres résolutions.

### Wrapper minimal recommandé

La forme ci-dessous décrit le découpage à implémenter. Elle n’est pas un extrait exécutable du dépôt upstream : déplacer les effets globaux de `swap.py` derrière `configure` est nécessaire.

```python
class ReSwapperBackend:
    def configure(self, source_path, model_path, emap_path, device):
        # Valider les fichiers et leur empreinte avant torch.load.
        # Charger FaceAnalysis une fois, puis prepare(ctx_id=device_id).
        # Charger StyleTransferModel().to(device), état validé, eval().
        # Détecter le visage source, calculer latent [1,512] et le mettre en cache.
        ...

    def process(self, frame_bgr):
        # Vérifier uint8/HWC/contigu. Retourner sans modification si aucune face.
        # Aligner le visage cible et garder la matrice affine M.
        # Construire le blob et appeler le modèle sous torch.inference_mode().
        # Reprojeter avec M et retourner une nouvelle frame BGR.
        ...
```

`swap.py` construit `FaceAnalysis(name='buffalo_l')` et exécute `prepare(ctx_id=0, det_size=(512,512))` au niveau module. Son import amorce donc le runtime modèle et choisit le GPU avant l’initialisation applicative. `Image.py` lit également `emap.npy` dans son corps de module. Un bridge robuste ne doit pas importer ces modules avant d’avoir contrôlé les assets et choisi son device. Il peut réutiliser la classe de réseau et les routines d’image tout en instanciant le détecteur explicitement dans `configure`.

`swap_face` renvoie un crop BGR; la fonction Torch convertit `target` et `source` vers le device à chaque appel. Cela marche fonctionnellement, mais conserve une copie CPU→GPU du latent à chaque frame. L’adaptateur pourra garder `source_latent` en tensor GPU et appeler directement le réseau ; ce serait une adaptation locale à mesurer et à valider après le premier bridge.

### Assets et démarrage hors réseau

Le worker doit vérifier avant initialisation :

- poids ReSwapper `.pth` choisis, hash SHA-256 attendu et chemin local ; le chargement `torch.load` lit un pickle, donc n’accepter que le fichier explicitement approuvé ;
- `emap.npy`, hash/périmètre de licence consignés ;
- paquet InsightFace `buffalo_l` présent dans le répertoire local avec ses fichiers de détection/embedding, provider choisi et modèle légalement utilisable ; échouer si l’auto-téléchargement serait nécessaire ;
- compatibilité CUDA/driver et disponibilité effective de `torch.cuda.is_available()` pour le device configuré.

Le modèle `reswapper-1019500.pth` et les alternatives viennent du dépôt HF `somanchiu/reswapper`; le modèle HF déclare AGPL-3.0. Le fichier `emap.npy` est décrit comme extrait d’inswapper. Les poids InsightFace ont des conditions distinctes du code, et les modèles publics InsightFace sont limités à la recherche non commerciale selon leur politique. Ne pas activer le worker en production tant que l’origine/licence/usage des trois groupes de fichiers ne sont pas établis.

## Worker audio Seed-VC realtime

### Importer le noyau sans GUI, VAD ni audio-device

Le fichier est nommé `real-time-gui.py`, donc ne peut pas être importé par un `import real-time-gui` ordinaire. Les fonctions `load_models` et `custom_infer` sont définies au niveau module. Le code FreeSimpleGUI, `sounddevice`, la classe `GUI` et son `AutoModel` FunASR sont sous `if __name__ == "__main__"`; importer le module sous un autre nom ne construit donc ni GUI, ni VAD, ni stream de périphérique.

L’import au niveau module charge toutefois `torch`, `torchaudio`, `librosa`, YAML, `modules.commons` et des utilitaires. Il suppose aussi que le processus part du checkout Seed-VC : le fichier ajoute le répertoire courant à `sys.path` et ouvre des chemins relatifs (`configs/...`). Le worker doit démarrer avec `cwd` égal à la racine de son checkout, puis charger le fichier par `importlib.util.spec_from_file_location`. Une alternative ultérieure est d’extraire ces deux fonctions vers un module Python au nom importable, en conservant l’attribution GPL et en gardant le comportement pinned.

`device` vaut initialement `None`; `main` lui attribue `cuda:<gpu>` avant d’appeler `GUI`. `load_models` ne fait pas cette attribution lui-même. Dans le worker, définir explicitement `module.device = torch.device("cuda:N")` avant l’appel. La callback de l’upstream prend la branche `torch.cuda.Event` pour tout device qui n’est pas MPS : le mode CPU ne constitue pas un fallback exploitable dans cette révision. Le mode visé ici est CUDA; échouer au démarrage si le GPU demandé n’est pas disponible.

### Chargement contrôlé et assets

Exécuter le processus avec les variables offline avant l’import de Torch/Transformers/Hugging Face :

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Sur Windows, les définir dans l’environnement du processus enfant avant son lancement. `HF_DATASETS_OFFLINE=1` peut aussi être fixé par défense, mais le chemin inference montré n’utilise pas directement `datasets`. Le mode offline rend les absences bloquantes ; il ne télécharge pas ce qui manque.

Créer les arguments attendus par `load_models` :

```python
args.checkpoint_path = r"D:\models\seed-vc\DiT_uvit_tat_xlsr_ema.pth"
args.config_path = r"D:\models\seed-vc\config_dit_mel_seed_uvit_xlsr_tiny.yml"
args.fp16 = True
```

Lorsque `checkpoint_path` est non vide, la première paire checkpoint/config peut être locale. `load_models` télécharge/charge encore des ressources via les helpers Hugging Face ou `from_pretrained`; prévoir et vérifier les assets correspondants avant d’activer offline :

- config YAML realtime `config_dit_mel_seed_uvit_xlsr_tiny.yml` (22 050 Hz, hop 256, XLS-R, vocodeur HiFi-GAN) ;
- checkpoint Seed-VC `DiT_uvit_tat_xlsr_ema.pth` ;
- modèle et feature extractor `facebook/wav2vec2-xls-r-300m` nommés dans la config ;
- `campplus_cn_common.bin` du dépôt `funasr/campplus` ;
- `hift.pt` du dépôt `FunAudioLLM/CosyVoice-300M`, ainsi que le `configs/hifigan.yml` local du checkout.

Les poids Seed-VC sont sous GPL-3.0 sur la fiche HF ; les ressources auxiliaires ont leur provenance/licence propres. Pour un vrai offline, les versions exactes de ces assets et toutes leurs empreintes doivent être enregistrées dans la configuration de déploiement, puis validées localement. Les chemins de cache HF ne sont pas à traiter comme une preuve d’intégrité : vérifier tailles, hashes attendus et révisions avant le lancement. Ne pas démarrer une résolution de modèle HF implicite si un chemin local requis est absent.

La callback GUI appelle aussi FunASR AutoModel pour le VAD. `custom_infer` ne dépend pas de ce VAD : l’intégration peut commencer sans le VAD upstream. Décider séparément si les blocs silencieux sont convertis ou transmis tels quels, au lieu d’instancier `AutoModel` accidentellement avec `GUI`.

### Appels stateful et paramètres exacts

`load_models(args)` renvoie un tuple `model_set` de six éléments : modèle DiT, fonction encodeur sémantique, fonction vocodeur, CAMPPlus, fonction mel et dictionnaire `mel_fn_args`. Pour la preset realtime du YAML, `mel_fn_args["sampling_rate"] == 22050` et `mel_fn_args["hop_size"] == 256`.

La fonction de bloc réellement appelable est :

```python
audio_tensor = module.custom_infer(
    model_set,
    reference_wav,             # numpy float mono au SR modèle, 22050 Hz
    reference_identity,        # clé stable; changement invalide les caches prompt
    input_wav_res,             # tensor mono historique déjà rééchantillonné à 16 kHz
    block_frame_16k,           # durée du nouveau bloc en samples 16 kHz
    skip_head,                 # contexte gauche en unités de 20 ms
    skip_tail,                 # contexte droit en unités de 20 ms
    return_length,             # longueur produite en unités de 20 ms
    diffusion_steps,
    inference_cfg_rate,
    max_prompt_length,
    cd_difference,             # contexte CE en excès sur le contexte DiT
)
```

La référence est mono float32, chargée/rééchantillonnée à 22,05 kHz. La première invocation (ou une référence/longueur de prompt modifiée) calcule les caractéristiques sémantiques et timbrales de la référence et les garde dans les globaux `prompt_condition`, `mel2`, `style2`, `reference_wav_name` et `prompt_len`. Les appels doivent rester sérialisés dans un seul worker et réutiliser le même module ; `custom_infer` n’est pas réentrant. Au changement de référence ou à un redémarrage logique, invalider les caches globaux ou recréer le worker.

La forme minimale d’un wrapper stateful est la suivante. Le pseudocode exprime le cycle de vie et le contrat; `append_context`, `resample_window` et `sola_join` représentent l’état de buffer et le raccord décrits ci-dessous, ils restent à implémenter et mesurer.

```python
class SeedRealtimeBackend:
    def start(self, seed_root, checkpoint, config, gpu_id, reference_path, params):
        # Le parent définit HF_HUB_OFFLINE et TRANSFORMERS_OFFLINE avant spawn.
        # cwd = seed_root; sys.path contient seed_root avant les imports.
        self.module = load_py_file_with_importlib(seed_root / "real-time-gui.py")
        self.module.device = torch.device(f"cuda:{gpu_id}")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU indisponible")
        args = Namespace(checkpoint_path=checkpoint, config_path=config, fp16=True)
        self.model_set = self.module.load_models(args)
        self.sr = self.model_set[-1]["sampling_rate"]
        self.reference, _ = librosa.load(reference_path, sr=self.sr, mono=True)
        self.ref_key = stable_local_reference_id(reference_path)
        self.params = params
        self.clear_audio_history_and_sola()

    def process_block(self, mono_float_block, input_sr):
        window = self.append_context(mono_float_block, input_sr)
        window_16k = self.resample_window(window, input_sr, 16000)
        converted_22050 = self.module.custom_infer(
            self.model_set, self.reference, self.ref_key, window_16k,
            self.block_frame_16k, self.skip_head, self.skip_tail,
            self.return_length, self.params.steps, self.params.cfg,
            self.params.prompt_seconds, self.params.ce_minus_dit_context)
        converted = self.resample_model_output(converted_22050, self.sr, input_sr)
        return self.sola_join(converted)  # exactly one block, preserving overlap state
```

Pour une même voix/réglage, le worker garde `module`, `model_set`, les buffers waveform/16 kHz, les paramètres de fenêtrage et l’overlap SOLA entre blocs. Éviter de recharger le checkpoint à chaque appel ou d’instancier `GUI`. La méthode doit traiter les blocs en ordre et refuser les changements de taux/paramètres qui impliquent une recréation de buffers sans `reset` explicite.

`input_wav_res` n’est pas le bloc isolé : c’est la fenêtre mono de 16 kHz incluant historique gauche, bloc courant et contexte droit. Sa taille et sa progression doivent rester alignées sur le buffer à la fréquence d’entrée. Le code upstream garde l’historique 22,05 kHz dans `input_wav`, le décale de `block_frame`, ajoute le nouveau bloc, puis rééchantillonne la fenêtre utile en 16 kHz et retire 320 samples (20 ms) de bord de resampling. `block_frame_16k` sert au décalage du buffer de features à 16 kHz.

Les durées sont converties en unités de 20 ms par `zc = samplerate // 50` (441 samples à 22,05 kHz) :

```text
block_frame        = round(block_time * samplerate / zc) * zc
block_frame_16k    = 320 * block_frame // zc
crossfade_frame    = round(crossfade_time * samplerate / zc) * zc
sola_buffer_frame  = min(crossfade_frame, 4 * zc)  # max 80 ms
sola_search_frame  = zc                            # 20 ms
extra_frame        = round(extra_time_ce * samplerate / zc) * zc
extra_frame_right  = round(extra_time_right * samplerate / zc) * zc
skip_head          = extra_frame // zc
skip_tail          = extra_frame_right // zc
return_length      = (block_frame + sola_buffer_frame + sola_search_frame) // zc
```

`custom_infer` retire `int(cd_difference*50)` frames au début des features, demande au régulateur la longueur correspondant à `skip_head + return_length + skip_tail` moins ce contexte CE, puis coupe la forme d’onde vocodée pour enlever le contexte droit et retourner `return_length * samplerate // 50` samples à 22,05 kHz. Son `block_frame_16k` est passé mais n’est pas consommé dans le corps de la fonction à ce commit. Ne pas enlever le contexte droit ni retailler la sortie une seconde fois dans le bridge.

### Resampling, SOLA et latence

Dans l’upstream, le bloc micro est réduit en mono, puis la fenêtre de contexte est convertie à 16 kHz pour l’encodeur XLS-R. Seed-VC produit à 22,05 kHz. Si la fréquence périphérique diffère du taux modèle, la callback rééchantillonne ensuite la sortie au taux du périphérique. Dans l’app, le worker peut garder une interface mono à 22,05 kHz pour simplifier : le service audio frontal s’occupe de resampler l’entrée/sortie, mais doit conserver des blocs de taille constante et le même historique logique.

SOLA (`Synchronized Overlap-Add`) raccorde les chunks successifs. Pour chaque résultat, le code compare les `sola_buffer_frame + sola_search_frame` premiers échantillons avec l’overlap gardé du chunk précédent, via corrélation normalisée. Il choisit un offset dans la zone de recherche de 20 ms, supprime cet offset, applique une fenêtre de crossfade en sinus carré de longueur `sola_buffer_frame`, garde la suite `[block_frame:block_frame+sola_buffer_frame]` comme overlap pour le prochain appel, puis émet exactement `block_frame` samples. Une réimplémentation doit préserver cet état entre appels ; appliquer un fondu simple sans détection d’offset risque de clics ou d’alignement instable.

La latence est le prix du lookahead et du traitement par blocs. La UI estime le délai affiché avec `stream.latency[-1] + block_time + crossfade_time + extra_time_right + 0.01`. Le README du projet exprime l’ordre de grandeur algorithmique comme environ `2 * block_time + extra_time_right`, auquel s’ajoute le délai device (souvent ~100 ms). Les nombres sont des estimations; mesurer p50/p95 sur le GPU cible en présence des autres charges réelles.

Ne pas copier les valeurs « défaut » de mémoire de la classe sans lire `configs/inuse/config.json` : la configuration initiale créée par le script est `block_time=0.3`, crossfade 0.04, `extra_time_ce=2.5`, `extra_time=0.5`, contexte droit 0.02, steps 10, CFG 0.7, prompt max 3s. `GUIConfig` a d’autres valeurs par défaut de secours, dont 2.0s de contexte droit. Pour commencer, prendre la configuration persistée (0.3/0.04/2.5/0.5/0.02) et la faire correspondre exactement aux paramètres exposés dans le worker, puis profiler ; les valeurs de mesure README sont un autre preset et ne doivent pas être mélangées implicitement.

Le code logge chaque forme de bloc, fait des synchronisations CUDA autour de l’inférence et écrit l’UI depuis le callback. Ces opérations doivent être retirées/redirectées dans le worker de production après validation ; la sortie IPC doit conserver timestamp/sequence, taux d’échantillonnage, nombre d’échantillons et état d’erreur.

## Sources locales exactes et vérification

Les téléchargements de sources publics demandés ont été placés sous `/tmp/unified-upstream` par `gh api`, sans poids de modèle : les 10 `.py` ReSwapper du commit choisi, le fichier complet `real-time-gui.py` et son preset YAML Seed-VC, plus les deux `requirements.txt`. `manifests/upstream-source-manifest.json` consigne les commits, dates, chemins d’origine et empreintes SHA-256. Aucun fichier `.onnx`, `.pth`, `.pt`, `.bin`, `.npy` ou `.safetensors` n’a été téléchargé.

Sources upstream :

- [ReSwapper commit](https://github.com/somanchiu/ReSwapper/tree/44474cb8d85d15274ae274e5e0f5f44f0814af51), notamment [swap.py](https://github.com/somanchiu/ReSwapper/blob/44474cb8d85d15274ae274e5e0f5f44f0814af51/swap.py) et [StyleTransferModel_128.py](https://github.com/somanchiu/ReSwapper/blob/44474cb8d85d15274ae274e5e0f5f44f0814af51/StyleTransferModel_128.py).
- [Seed-VC commit](https://github.com/Plachtaa/seed-vc/tree/51383efd921027683c89e5348211d93ff12ac2a8), notamment [real-time-gui.py](https://github.com/Plachtaa/seed-vc/blob/51383efd921027683c89e5348211d93ff12ac2a8/real-time-gui.py) et [le preset XLSR realtime](https://github.com/Plachtaa/seed-vc/blob/51383efd921027683c89e5348211d93ff12ac2a8/configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml).
