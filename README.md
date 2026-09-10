# 🚗 Système de Détection, Tracking et Comptage de Véhicules avec YOLOv11 & BoT-SORT

Ce projet met en œuvre une solution intelligente d'analyse vidéo pour la détection, le suivi (tracking) et l'analyse du temps de séjour de véhicules (voitures, bus, camions) traversant une Zone d'Intérêt (ROI - Region of Interest) prédéfinie.

---

## 🛠️ Instructions d'Installation et d’Exécution

### Prérequis
- **Python** 3.13 ou supérieur
- Gestionnaire de paquets **`uv`** (recommandé pour une gestion rapide des environnements et dépendances)

### 1. Clonage et Installation des Dépendances

Clonez le projet et installez l'environnement virtuel ainsi que les dépendances avec `uv` :

```bash
# Se placer dans le répertoire du projet
cd Yolo_project

# Synchroniser l'environnement et installer les dépendances
uv sync
```

*Si vous souhaitez réinstaller manuellement les bibliothèques requises :*
```bash
uv add "opencv-python>=5.0.0.93" "pandas>=3.0.5" "ultralytics>=8.4.146"
```

### 2. Exécution du Script Principal

Pour exécuter le script de détection et de tracking sur la vidéo `vid.mp4` :

```bash
uv run main.py
```

### 3. Fonctionnement du Script (`main.py`)
- Charge le modèle de détection **YOLOv11** (`yolo11n.pt`).
- Utilise l'algorithme de suivi **BoT-SORT** configuré via `botsort_reid.yaml`.
- Définit un polygone de zone d'intérêt (Zone A) sur l'image.
- Suit l'entrée des véhicules dans la zone, journalise les sorties et calcule la durée exacte d'arrêt/présence dans la zone.
- Exporte l'ensemble des événements au format JSON dans `events.json`.
- Enregistre la vidéo annotée finale dans `video_annotated.mp4`.

---

## 📦 Bibliothèques et Versions Utilisées

Les dépendances majeures du projet sont gérées via `pyproject.toml` et l'environnement `uv` :

| Bibliothèque | Version minimale / utilisée | Rôle / Utilisation |
| :--- | :--- | :--- |
| **Python** | `>= 3.13` | Environnement d'exécution principal |
| **`ultralytics`** | `>= 8.4.146` | Détection d'objets (YOLOv11) et pipeline de tracking (BoT-SORT / ByteTrack) |
| **`opencv-python`** | `>= 5.0.0.93` | Traitement d'images, dessin des ROI/polygones, gestion des flux vidéo (`VideoCapture`, `VideoWriter`) |
| **`pandas`** | `>= 3.0.5` | Manipulation et structuration avancée des données analytiques |
| **`numpy`** | N/A (Dépendance core) | Opérations matricielles et géométrie vectorielle du polygone ROI |
| **`json`** | Standard Library | Exportation structurée de l'historique des événements de passage |

---

## 🎬 Démonstration des Résultats et Outputs

### 1. Sortie Console (Terminal)
Pendant l'exécution, le système affiche les statistiques d'inférence en temps réel ainsi que les événements de franchissement de la zone d'intérêt :

```text
0: 384x640 22 cars, 219.4ms
Speed: 13.3ms preprocess, 219.4ms inference, 35.6ms postprocess per image at shape (1, 3, 384, 640)
 Vehicle ID 1 entered
 Vehicle ID 2 entered
 Vehicle ID 3 entered
 ...
 Vehicle ID 22 entered

0: 384x640 21 cars, 207.1ms
Speed: 11.3ms preprocess, 207.1ms inference, 2.6ms postprocess per image at shape (1, 3, 384, 640)
 Vehicle ID 25 entered
Vehicle ID 17 exited
Vehicle ID 17 stayed in zone for 0.07 seconds
Vehicle ID 20 exited
Vehicle ID 20 stayed in zone for 0.07 seconds
```

### 2. Journal d'Événements (`events.json`)
Chaque entrée ou sortie de zone génère un objet JSON horodaté :

```json
[
    {
        "event": "vehicle_entering_the_zone",
        "track_id": 1,
        "zone": "Zone A",
        "timestamp": 0.03,
        "confidence": 0.6024
    },
    {
        "event": "vehicle_exiting_the_zone",
        "track_id": 17,
        "zone": "Zone A",
        "timestamp": 0.13
    }
]
```

### 3. Fichier Vidéo Enregistré (`video_annotated.mp4`)
Un fichier vidéo récapitulatif est produit à la racine du projet (`video_annotated.mp4`), affichant la zone polygonale (polylignes vertes), les boîtes englobantes (*bounding boxes*), les identifiants de suivi (*Track IDs*) et la classe du véhicule.

---

## 📐 Métrique Principale Utilisée : IoU (Intersection over Union)

La métrique centrale utilisée pour mesurer le chevauchement spatial et associer les véhicules entre deux frames consécutives est **l'Intersection over Union (IoU)**.

### 1. Formule et Principe
L'IoU mesure le niveau de superposition entre deux boîtes englobantes (*Bounding Boxes* $B_1$ et $B_2$) :

$$\text{IoU} = \frac{\text{Aire de l'Intersection}(B_1 \cap B_2)}{\text{Aire de l'Union}(B_1 \cup B_2)}$$

- **Implémentation dans le code** : La fonction utilitaire [`calculate_iou(box1, box2)`](file:///c:/Users/hp/Desktop/Yolo_project/main.py#L32-L52) est définie dans [`main.py`](file:///c:/Users/hp/Desktop/Yolo_project/main.py).
- **Fonctionnement dans le pipeline** : L'algorithme de suivi BoT-SORT (via [`botsort_reid.yaml`](file:///c:/Users/hp/Desktop/Yolo_project/botsort_reid.yaml)) calcule l'IoU automatiquement en arrière-plan à chaque frame pour associer les boîtes englobantes et valider les trajectoires. L'output console principal de `main.py` affiche les événements de franchissement (`Vehicle ID entered / exited`), tandis que l'IoU sert de critère d'association interne.

### 2. Valeurs d'IoU Mesurées sur le Flux Vidéo du Projet

Lors d'une évaluation directe avec la fonction `calculate_iou` sur les frames successives de `vid.mp4`, nous obtenons les statistiques réelles suivantes :

| Indicateur IoU | Valeur Mesurée | Interprétation pour notre Projet |
| :--- | :--- | :--- |
| **IoU Moyen** | **`0.8716` (87.16%)** | **Haute Stabilité** : La grande majorité des véhicules conservent une superposition d'environ 87% entre deux images successives. |
| **IoU Médian** | **`0.9070` (90.70%)** | **Excellente Précision** : Plus de 50% des associations dépassent 90% de superposition d'image en image. |
| **IoU Maximum** | **`0.9964` (99.64%)** | **Quasi-Statique** : Obtenu pour les véhicules immobiles ou en circulation très lente. |
| **IoU Minimum** | **`0.5432` (54.32%)** | **Mouvement Rapide / Virages** : Correspond aux véhicules accélérant ou modifiant rapidement leur trajectoire. |

### 3. Grille de Seuil et Interprétation

| Valeur d'IoU | Interprétation et Résultat dans le Projet |
| :--- | :--- |
| **$\text{IoU} = 1.0$** | **Superposition Parfaite** : Les deux boîtes englobantes coïncident parfaitement. |
| **$\text{IoU} \ge 0.8$** | **Association Réussie (`match_thresh: 0.8`)** : Le véhicule conserve son même *Track ID* d'une image à la suivante avec une haute certitude. |
| **$0.5 \le \text{IoU} < 0.8$** | **Proximité Modérée (`proximity_thresh: 0.5`)** : Indique un véhicule proche ou en mouvement rapide entre deux images. |
| **$\text{IoU} < 0.25$** | **Seuil de Dissociation (`track_high_thresh: 0.25`)** : En-dessous de ce niveau, la correspondance est rejetée et le système peut initialiser un nouveau véhicule. |
| **$\text{IoU} = 0.0$** | **Aucun Chevauchement** : Aucune aire commune entre les deux véhicules. |

---

## ⚠️ Principaux Failure Modes Identifiés

Lors de l'analyse automatique du trafic, plusieurs scénarios complexes peuvent dégrader les performances de détection et de suivi :

| Failure Mode | Observation | Impact mesuré |
| :--- | :--- | :--- |
| **Occlusion** | Un véhicule est partiellement caché par un autre véhicule (camion, bus) ou un obstacle fixe. | Perte temporaire du suivi ou changement d'ID (*ID Switch*). |
| **ID Switch (IDSW)** | Deux véhicules très proches ou se croisant échangent leurs Track IDs. | Augmentation du nombre de commutations d'ID (*IDSW*). |
| **Forte proximité / croisement** | Les boîtes englobantes (*bounding boxes*) de deux véhicules se superposent fortement. | |
| **Variation d'éclairage** | Passage d'une zone ensoleillée à une zone d'ombre ou baisse globale de luminosité. | |
| **Qualité vidéo dégradée** | Flou de mouvement (*motion blur*) ou véhicules trop éloignés/petits dans la scène. | |
| **Perte puis reprise du suivi** | Un véhicule disparaît momentanément (ex. sous un pont/ombre) puis réapparaît. | |

---

## 🚀 Propositions d'Amélioration & Méthode de Mesure d'Impact

### Proposition 1 : Activation du modèle de Re-Identification (ReID) d'apparence dans BoT-SORT

* **Description de l'amélioration** :
  Actuellement, le tracker BoT-SORT est exécuté avec l'option `with_reid: False` dans le fichier `botsort_reid.yaml`. En activant le module ReID (`with_reid: True`) et en fournissant un modèle d'extraction de descripteurs visuels (ex. OSNet ou ResNet entrainé sur des véhicules), l'algorithme combine la trajectoire (filtre de Kalman) **et** la signature visuelle du véhicule.

* **Méthode de mesure de l'impact** :
  1. **Dataset d'évaluation** : Constituer ou utiliser un jeu de test annoté avec vérités terrain (*Ground Truth*) contenant des occlusions et des croisements.
  2. **Métriques à mesurer (CLEAR MOT & HOTA)** :
     - **IDSW (Identity Switches)** : Compter le nombre total d'inversions d'identifiants. *Objectif : Réduction de 30% à 50% des IDSW.*
     - **IDF1 Score** : Mesurer la capacité du tracker à maintenir une identité constante sur l'ensemble de la trajectoire. *Objectif : Hausse de l'IDF1.*
     - **MOTA (Multiple Object Tracking Accuracy)** : Vérifier que l'association globale reste robuste.

---

### Proposition 2 : Fine-Tuning du Détecteur YOLOv11 sur un Jeu de Données de Trafic Routier (UA-DETRAC / VisDrone)

* **Description de l'amélioration** :
  Re-traîner ou ajuster le modèle pré-entraîné COCO (`yolo11n.pt`) sur un dataset spécialisé en surveillance routière (ex. UA-DETRAC, VisDrone, ou BDD100K). Cela permet au réseau de mieux détecter les petits véhicules en arrière-plan et de tolérer le flou de mouvement ainsi que les variations de luminosité.

* **Méthode de mesure de l'impact** :
  1. **Métriques de Détection (mAP & Recall)** :
     - **mAP@0.5 & mAP@0.5:0.95** : Évaluer la précision des bounding boxes sur le jeu de test de véhicules.
     - **Recall & Taux de Faux Négatifs (FN)** : Valider que le nombre de véhicules manqués (particulièrement de petite taille) diminue significativement.
  2. **Mesure de l'impact sur le Tracking** :
     - Calcul du ratio $\text{FN} / \text{Nombre total d'objets}$ avant et après fine-tuning pour quantifier la hausse de sensibilité.