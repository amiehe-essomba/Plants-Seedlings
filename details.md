#### *```Comment extraire le dataset en une ligne de commande avec read_plant_datasets()```*

* *```Difinir le chemin (path)```*
    ```python
    L argument path permet de difinir la localisation du répertoire contenant les espèces.
    type(path) = <class str>
    path = "my_path"
    ```
* *```Choisir les indices```*
    ```python
    type_indexes est un argument permettant de sélectionner les répertoires à extraire 
    type(type_indexes) = <lass list>
    
    samples         = 12 # nombre total d'espèces différentes de plantes
    pas             = 1  # le pas 
    type_indexes    = [x for x in range(0, samples, pas) ]

    Si samples = 12 et pas = 1 tout le dataset est extrait et les fichiers sont catégorisés en "samples" classes différentes.
    Sinon on peut changer les paramètres "samples" et "pas" pour choisir des répertoies spécifiques
    ```

* *```Choisir un type de filtrage```*
    ```python
    channel_type est un argument permettant de spécifier le type de filtre à appliquer sur les images lors de l extraction.
    type(channel_type) = <class str>

    à savoir Il exite plusieurs types de filtres implémentés ici. Il faut en choisir un filtre dans la liste ci-dessous
    ["RBG-HSV", "SIMPLE_GRAY", "RGB", "HIS-C", "HIS-EQ","HIS-ADAPT", "HIS-DISK", "GAUSSIAN", "RGR2-HSV", "RGR2-LAB" ]
    
    channel_type        = "RGR2-LAB" 
    ```

* *```redimensionnement des images```*
    ```python
    reshape est un argument permettant de redimenionner les images en plusieurs tailles différentes:
    type(reshape) = <class : list>

    reshape = [(160, 160), (300, 300)] # pour un redimensionnement en 2 tailles différentes 160x160 et 300x300

    ```

* *```Choisir le format de sorti```*
    ```python
    return_as est un argument qui spéficie le format
    type(return_as) = <class str>

    si return_as = "dict" le format est dictionaire avec comme schema :
    data = {    
    'X'             : X,                            # images originales
    'images'        : true_imgs,                    # images + channel_type
    "target"        : y,                            # cible
    "feature_names" : feature_names,                # noms des espèces
    'subset'        : subset,                       # type de données
    "channel_type"  : channel_type,                 # RGG2-LAB
    "number_of_images"  : number_of_imgames,        # nombre d'images par espèces
    "shape"             : reshape,                  # noyau de redim
    "sobels"        : sobels,                       # nombre de canaux RGBA par espèces 
    "rapport"       : rapport,                      # largeur/hauteur 
    "pixels"        : pixels,                       # (largaur * hauteur) / 1024
    "width"         : widths,                       # largeurs des images
    "height"        : heights                       # hauteurs de images ,
    "paths"         : paths                         # locations de chaque images
    } 

    si return_as = "X_y" le format est un tuple or (les images, et la cible sont des sorties.
    ```
* *```Progression```*
    ```python
    verbose permet d imprimer la progression de l extraction dans le temps 

    verbose  = 1 : True 
    verbose != 1 : False
    ```
    