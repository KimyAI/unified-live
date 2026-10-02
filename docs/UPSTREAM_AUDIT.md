# Audit des sources upstream pour Unified Live

État de la recherche : 2 octobre 2026. Les références de code ci-dessous sont épinglées aux commits quand ils ont été fournis ou vérifiés ; les autres dépôts sont examinés à leur tête publique indiquée. Cet audit distingue licence du code, licence/statut des poids et faisabilité technique. Ce n’est pas un avis juridique.

## Résultat pratique

Pour le visage, ReSwapper expose le point de calcul le plus simple à réutiliser par frame, sans caméra ni fenêtre : chargement du modèle, création d’un crop aligné, calcul d’un latent source, `swap_face`, puis collage inverse. Une petite couche applicative autour de ces opérations est nécessaire, car le dépôt ne fournit pas de classe service ou d’API de streaming. L’import de `swap.py` initialise toutefois InsightFace et le GPU globalement, et le script de démonstration est orienté fichiers. DoppleDanger confirme que cette chaîne fonctionne en direct, mais son point d’entrée boucle sur webcam/OpenCV et ses fenêtres.

Pour Seed-VC realtime, le fichier `real-time-gui.py` contient une vraie fonction d’inférence par blocs, `custom_infer(...)`. C’est le meilleur noyau à encapsuler dans un sous-processus CUDA : charger les modèles et la référence une seule fois, recevoir les blocs PCM par IPC, réutiliser le contexte et buffers, puis renvoyer l’audio. Le script lui-même est attaché à FreeSimpleGUI, `sounddevice`, au choix de devices et à un VAD. `SeedVCWrapper.convert_voice(...)` est une voie batch sur fichiers, pas une API d’audio live. Le dépôt Seed-VC a été archivé le 21 novembre 2025.

Les licences de modèles empêchent de conclure que « code open source » signifie « poids utilisables librement ». En particulier, les modèles publics InsightFace utilisés pour détection/embeddings sont réservés à la recherche académique non commerciale ; InsightFace demande une autorisation distincte pour ses modèles de face swap. Le dépôt de poids ReSwapper déclare AGPL-3.0, mais le README indique que `emap.npy` provient du modèle inswapper original : l’autorisation d’usage des poids et de leurs éléments dérivés doit être traitée séparément.

## Dépôts, révisions et licences

| Dépôt | Révision examinée | Code | Poids et données |
|---|---|---|---|
| [somanchiu/ReSwapper](https://github.com/somanchiu/ReSwapper/tree/44474cb8d85d15274ae274e5e0f5f44f0814af51) | `44474cb8d85d15274ae274e5e0f5f44f0814af51`, 2025-06-14 05:32 UTC | AGPL-3.0 | [HF somanchiu/reswapper](https://huggingface.co/somanchiu/reswapper) déclare AGPL-3.0. `emap.npy` est indiqué comme extrait d’inswapper ; consulter la licence InsightFace et celle du modèle avant redistribution ou usage commercial. |
| [hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam/tree/759e3f985811985cf6c00a3e43e9579652c318f5) | `759e3f985811985cf6c00a3e43e9579652c318f5`, 2026-09-26 10:05 UTC | AGPL-3.0 | Les fichiers inswapper sont externes au code. La documentation officielle InsightFace réserve ses modèles publics à la recherche non commerciale et demande de contacter InsightFace pour les modèles de face swap. |
| [Tantalum-Labs/DoppleDanger](https://github.com/Tantalum-Labs/DoppleDanger/tree/a7e09917680b33ddf667a084d9ea99fd1633b121) (ancien luispark6) | `a7e09917680b33ddf667a084d9ea99fd1633b121`, 2025-08-07 22:08 UTC | AGPL-3.0 à la racine ; le dossier intégré `seed_vc/` contient une licence GPL-3.0 | ReSwapper `.pth`, GFPGAN et modèles Seed-VC sont des fichiers externes. Il faut vérifier les conditions de chacun ; le README ne confère pas à lui seul les droits de redistribution. |
| [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc/tree/51383efd921027683c89e5348211d93ff12ac2a8) | `51383efd921027683c89e5348211d93ff12ac2a8`, 2025-04-20 05:27 UTC ; dépôt archivé le 2025-11-21 | GPL-3.0 | [HF Plachta/Seed-VC](https://huggingface.co/Plachta/Seed-VC) déclare GPL-3.0. Les encodeurs/vocodeurs téléchargés (Whisper, CAM++, BigVGAN, etc.) ont leurs propres sources et licences à contrôler. |
| [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/81eed5e8f68b6bed1789f682fe78cdd324495afc) | `81eed5e8f68b6bed1789f682fe78cdd324495afc`, 2026-08-04 07:47 UTC | MIT pour le dépôt principal ; lire aussi le fichier de conditions et avis des dépendances à la racine | Les checkpoints et index voix sont externes et doivent être évalués par provenance/modèle. Le README renvoie aux poids Hugging Face séparément du code. |
| [w-okada/voice-changer](https://github.com/w-okada/voice-changer/tree/d8ef15799470193f7c8176ef471245753a656626) | `d8ef15799470193f7c8176ef471245753a656626`, 2026-09-26 20:32 UTC | Le `LICENSE` racine est MIT avec plusieurs notices MIT intégrées ; l’API de modèles n’est pas une garantie de licence uniforme | Les modèles et corpus ont des conditions indépendantes. Le README mentionne notamment Beatrice JVS comme exception non MIT et limitée à Windows/CPU pour v1. |

Les déclarations de licence HF sont celles visibles sur les fiches de modèle au moment de la consultation. Pour les poids non hébergés avec une licence explicite ou pour des sorties dérivées, l’état reste à confirmer auprès de l’auteur concerné.

## ReSwapper : API de frame et limites

Fichiers à consulter dans le commit épinglé : [`swap.py`](https://github.com/somanchiu/ReSwapper/blob/44474cb8d85d15274ae274e5e0f5f44f0814af51/swap.py), [`Image.py`](https://github.com/somanchiu/ReSwapper/blob/44474cb8d85d15274ae274e5e0f5f44f0814af51/Image.py), [`StyleTransferModel_128.py`](https://github.com/somanchiu/ReSwapper/blob/44474cb8d85d15274ae274e5e0f5f44f0814af51/StyleTransferModel_128.py) et le [README](https://github.com/somanchiu/ReSwapper/blob/44474cb8d85d15274ae274e5e0f5f44f0814af51/README.md).

Les fonctions réellement appelables sont :

- `load_model(model_path)` : instancie `StyleTransferModel`, charge le `state_dict` PyTorch, passe en évaluation.
- `create_source(source_img_path)` : détecte le visage source, lit son embedding ArcFace normalisé, le projette avec `emap` et normalise le latent `[1,512]`. Retourne `None` si aucun visage n’est détecté.
- `create_target(target_image, resolution)` : détecte le visage cible, aligne selon ses cinq points, produit le tenseur blob cible et la matrice affine `M`. La version commitée accède directement à `[0]` et échoue si aucune cible n’est détectée.
- `swap_face(model, target_face, source_face_latent)` : exécute l’inférence du réseau PyTorch sans gradient et rend le crop BGR.
- `Image.blend_swapped_image(swapped_face, target_image, M)` : reprojette et mélange le crop dans l’image originale.

Contrat d’entrée documenté : cible RGB alignée `[1,3,128,128]` dans `[0,1]`; latent ArcFace `[1,512]`. La résolution 128 est le chemin le mieux défini ; README décrit une erreur d’alignement à d’autres résolutions et la correction approximative nécessaire. Les checkpoints disponibles sont `.pth` et certains `.onnx`, mais le code courant de `swap.py` charge le modèle PyTorch. README affirme qu’un ONNX exporté peut servir avec la classe INSwapper originale ; cette classe n’est pas implémentée dans ce dépôt. Éviter de confondre l’ONNX d’entraînement/export avec un runtime fourni.

Pour un worker isolé : ne pas importer le module depuis le processus UI sans étude du coût d’import. `swap.py` construit `FaceAnalysis(name='buffalo_l')` et appelle `prepare(ctx_id=0, det_size=(512,512))` au niveau module : cela amorce un provider CUDA et peut télécharger les modèles InsightFace. Déplacer l’initialisation dans le worker, charger les références/modèles une fois, accepter des images BGR déjà capturées depuis l’IPC, retourner image/erreur et traiter explicitement « aucun visage ». Aucune caméra n’est nécessaire aux quatre fonctions d’inférence ; `swap_live_video.py` de DoppleDanger, lui, ouvre `VideoCapture(0)`, des fenêtres OpenCV et éventuellement OBS.

Le chiffre d’environ 20 fps sur RTX 3090 vient du README de DoppleDanger, qui compare son assemblage au Deep-Live-Cam ; le README ReSwapper n’en fait pas état. Il s’agit d’une mesure rapportée par l’auteur de DoppleDanger, pas d’un benchmark indépendant. La détection InsightFace par frame, la copie GPU/CPU, l’alignement et le collage font partie du coût total.

## Seed-VC : chemin temps réel isolable

Dans le dépôt Seed-VC épinglé, [`real-time-gui.py`](https://github.com/Plachtaa/seed-vc/blob/51383efd921027683c89e5348211d93ff12ac2a8/real-time-gui.py) contient deux composants séparables :

- `load_models(args)` charge le modèle DiT temps réel, encodeur sémantique, vocodeur et modèle CAMPPlus, puis prépare les caches sur le device.
- `custom_infer(model_set, reference_wav, new_reference_wav_name, input_wav_res, block_frame_16k, skip_head, skip_tail, return_length, diffusion_steps, inference_cfg_rate, max_prompt_length, cd_difference=...)` met en cache les caractéristiques de la référence, encode le bloc d’entrée, construit le conditionnement, exécute le modèle et retourne un tenseur audio converti.

`custom_infer` dépend d’états globaux (`device`, `fp16`, prompt conditionnel, mél et style). Il faut garder ces appels dans un seul worker séquentiel ou rendre cet état explicite. `GUI.audio_callback` n’est pas un backend neutre : il utilise `sounddevice`, resampling, VAD FunASR, état audio, synchronisation CUDA/MPS, algorithme de crossfade/SOLA et `sd.Stream`. La création des devices/UI arrive dans la classe `GUI`. Réutiliser `custom_infer` dans un worker PCM permet d’éviter l’ouverture de micro, haut-parleur ou GUI par Seed-VC ; l’IPC et l’adaptation de blocs restent à implémenter.

`SeedVCWrapper` est une autre interface upstream mais son `convert_voice(source_path, target_path, ...)` charge des chemins audio et génère un résultat batch/stream de sortie pour la durée du fichier ; ce n’est pas le convertisseur de microphone par petits blocs. L’utiliser dans le chemin live ferait perdre la gestion d’état temporel du script realtime.

Le modèle présenté par le README comme adapté au temps réel est `seed-uvit-tat-xlsr-tiny` à 22,05 kHz. Le README donne un essai sur RTX 3060 Laptop : blocs de 180 ms, inférence de 150 ms, délai de 430 ms (temps algorithmique et périphérique). Ce sont des chiffres de l’auteur, dépendants du GPU, des paramètres et de la contention. Le code de la callback crée des `torch.cuda.Event` pour toute plateforme autre que MPS : la branche CPU-only ne fonctionne donc pas telle quelle. Les recommandations upstream sont d’avoir une carte GPU et de garder le temps d’inférence inférieur à la durée du bloc.

## Autres candidats

### Deep-Live-Cam

[`modules/processors/frame/face_swapper.py`](https://github.com/hacksider/Deep-Live-Cam/blob/759e3f985811985cf6c00a3e43e9579652c318f5/modules/processors/frame/face_swapper.py) fournit `pre_start()`, `get_face_swapper()`, `swap_face(source_face, target_face, temp_frame)` et `process_frame(source_face, temp_frame, target_face=None)`. `process_frame` traite une image et accepte un visage cible pré-calculé ; il n’ouvre pas la caméra à lui seul. Les fonctions reposent néanmoins sur le singleton `FACE_SWAPPER`, l’analyse InsightFace partagée, `modules.globals`, des providers configurés globalement et des post-traitements. C’est une intégration plus couplée que ReSwapper.

Les poids standard `inswapper_128.onnx` et `buffalo_l` doivent rester externalisés et soumis aux conditions InsightFace. Deep-Live-Cam lui-même est AGPL-3.0 ; incorporation de son code et distribution de l’application doit tenir compte des obligations AGPL.

### DoppleDanger

DoppleDanger prouve l’orchestration ReSwapper + Seed-VC et décrit le streaming caméra/micro/VB-CABLE/OBS. Côté visage, `create_source_latent`, `swap_face`, `face_align.norm_crop2`, `Image.getBlob`, `Image.blend_swapped_image_gpu` forment le chemin frame ; `swap_live_video.py` inclut le wrapper webcam et interface OpenCV. Côté voix, le dépôt embarque une copie de Seed-VC ; la racine AGPL-3.0 et la sous-licence Seed-VC GPL-3.0 rendent la réutilisation conjointe sous GPL à considérer, en plus des poids externes.

Le README demande Windows, Python 3.10, ffmpeg, CUDA 12.x et cuDNN 9.x. Il donne des commandes Torch 2.5.1+cu121, ONNX Runtime GPU 1.20, numpy 1.26.4 et installation de `requirements.txt --no-deps`. Cette dernière liste est volumineuse et inclut notamment cuPy CUDA 12.x, InsightFace, Torch/audio, FunASR, FreeSimpleGUI et PyQt6 ; le contournement `--no-deps` indique que le résolveur n’est pas une installation proprement reproductible.

### RVC WebUI

Le noyau d’inférence realtime est `infer.rtrvc.RVC`. Il accepte un modèle `.pth`, éventuellement un index FAISS, paramètres de pitch/formant et config device ; la méthode `RVC.infer(input_wav, block_frame_16k, skip_head, return_length, f0method)` convertit le segment tensoriel et retourne l’audio inféré. `realtime_gui.py` fournit le buffer/callback, mais charge son GUI FreeSimpleGUI et ouvre `sounddevice.Stream`. RVC convient plutôt à une voix entraînée/modèle local qu’à la conversion zero-shot Seed-VC.

À la révision examinée, trois manifests d’environnement sont présents : CPU/DirectML pour Windows, CUDA 11.8 et CUDA 12.8, ainsi qu’une cible Python 3.12. Les versions fournies avec le checkout actuel diffèrent fortement de celles ReSwapper/DoppleDanger (Torch 2.4.x/cu121 vs options récentes RVC Torch 2.7.1/cu118 ou cu128), même si les manifests RVC admettent des plages de versions d’ORT et NumPy. Ne pas fusionner leurs `requirements` par simple installation dans le même venv.

### w-okada/voice-changer

Le dépôt est une application serveur/client temps réel complète. La voie d’intégration backend la plus tangible est `VoiceChangerManager.changeVoice(unpackedData)` ; la route REST `MMVC_Rest_VoiceChanger.test` accepte un timestamp et un bloc audio base64, et retourne le bloc converti. Le serveur possède son propre `ServerDevice`, `VoiceChanger`, model slots, choix GPU, configuration et threads. Le réutiliser comme sous-processus/service évite d’ouvrir son client GUI dans l’app, mais nécessite démarrer son serveur et utiliser son protocole audio ; ce n’est pas une petite bibliothèque indépendante.

Le code global est principalement MIT, mais poids et modèles sont soumis à des conditions qui varient par modèle. Le README exclut explicitement Beatrice JVS de la licence MIT globale et en limite le support à Windows/CPU (v1). La diversité des backends et poids impose de sélectionner précisément le modèle si ce dépôt devient dépendance.

## Tensions de dépendances Windows/CUDA observées

- **ReSwapper main** épingle `torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`, `numpy==1.26.4`, `onnxruntime==1.18.1` et `onnxruntime-gpu==1.19.2` simultanément. Les distributions CPU et GPU d’ONNX Runtime occupent le même module Python ; cette paire ne doit pas être supposée saine. Le README réinstalle ensuite `torch` et `onnxruntime-gpu` à partir de wheels CUDA 12.1.
- **DoppleDanger** recommande `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `torchaudio==2.5.1+cu121`, `onnxruntime-gpu==1.20.0` et `numpy==1.26.4`, distincts des pins ReSwapper. Ses instructions suppriment les dépendances automatiques avec `--no-deps` puis ignorent les avertissements ; prévoir un environnement isolé ou reconstruire un lock contrôlé.
- **Seed-VC** propose dans le même requirements des index PyTorch nightly CUDA 12.6 puis des pins Torch 2.4.0 / torchvision 0.19.0 / torchaudio 2.4.0 sans suffixe CUDA explicite. Le chemin V2 `torch.compile` mentionne `triton-windows==3.2.0.post13` comme option Windows ; ne pas en faire une dépendance realtime v1 sans besoin validé.
- **RVC** publie des manifests CPU/DirectML, CUDA 11.8 et CUDA 12.8 distincts. Sa révision actuelle vise des versions récentes (instructions comments Torch 2.7.1) et ses ranges ORT diffèrent par CUDA target. Choisir exactement un manifeste/provider pour Windows.
- **InsightFace/ONNX Runtime GPU** lie le provider CUDA à des DLL/runtime CUDA compatibles. Le fait que les packages soient tous étiquetés « CUDA 12 » ne suffit pas à rendre les versions interchangeables ; le couple runtime/driver et provider de chaque worker doit être vérifié sur la machine cible.

Conclusion d’architecture : conserver deux environnements de sous-processus distincts au minimum (vision PyTorch/InsightFace et audio Seed-VC/torchaudio/FunASR), chacun recevant et renvoyant des buffers via IPC. Cela isole les imports, versions de Torch, providers CUDA et l’état global du realtime sans obliger l’app principale à démarrer une UI ou à réserver les périphériques audio/vidéo. Cette recommandation est une inférence des dépendances et points d’entrée ci-dessus ; elle n’a pas encore été mesurée sur la machine cible.

## Sources primaires consultées

- [ReSwapper README/code/licence](https://github.com/somanchiu/ReSwapper/tree/44474cb8d85d15274ae274e5e0f5f44f0814af51), [poids ReSwapper](https://huggingface.co/somanchiu/reswapper)
- [Deep-Live-Cam code/licence](https://github.com/hacksider/Deep-Live-Cam/tree/759e3f985811985cf6c00a3e43e9579652c318f5)
- [DoppleDanger README/code/licence](https://github.com/Tantalum-Labs/DoppleDanger/tree/a7e09917680b33ddf667a084d9ea99fd1633b121)
- [Seed-VC README/code/licence](https://github.com/Plachtaa/seed-vc/tree/51383efd921027683c89e5348211d93ff12ac2a8), [poids Seed-VC](https://huggingface.co/Plachta/Seed-VC)
- [InsightFace model licensing](https://github.com/deepinsight/insightface#license), [model zoo guidance](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md)
- [RVC WebUI README/code/licence](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/81eed5e8f68b6bed1789f682fe78cdd324495afc)
- [w-okada voice-changer README/code/licence](https://github.com/w-okada/voice-changer/tree/d8ef15799470193f7c8176ef471245753a656626)
