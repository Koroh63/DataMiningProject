from sklearn.preprocessing import MinMaxScaler, scale, StandardScaler, MaxAbsScaler, RobustScaler

# Redimension des données entre 0 et 1
def normalizeMinMax(dataset):
    scaler = MinMaxScaler();
    scaler.fit(dataset);
    return scaler.transform(dataset);

# Redimension des données en divisant par la valeur absolue maximale de chaque caractéristique entre -1 et 1
def normalizeMaxAbs(dataset):
    scaler = MaxAbsScaler();
    scaler.fit(dataset);
    return scaler.transform(dataset);

# Utilise les statistiques robustes pour la mise à l'échelle
def normalizeRobust(dataset):
    scaler = RobustScaler();
    scaler.fit(dataset);
    return scaler.transform(dataset);

# Normalisation d'une image entre 0 et 1
def normalizeImg(image):
    return image / 255.0

# Ajustement à une moyenne de 0 et une variance de 1
def standardisationZScore(dataset):
    scaler = StandardScaler();
    scaler.fit(dataset);
    return scaler.transform(dataset);