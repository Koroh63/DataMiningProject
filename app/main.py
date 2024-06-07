from src.model.load import loadCSVBase
from src.model import norm_stand as norm

if __name__ == "__main__":
    #baseView()
    dataset = loadCSVBase('./trees.csv');
    print(norm.normalizeMinMax(dataset));
    print(norm.normalizeMaxAbs(dataset));
    print(norm.normalizeRobust(dataset));
    print(norm.standardisationZScore(dataset));
    #norm.normalizeImg(img);