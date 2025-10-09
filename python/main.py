from sklearn.discriminant_analysis import StandardScaler
from kmeans_parallel import KMeans
import pandas as pd
import datetime

def main():
    train_path = "../data/movielens1m.csv"
    test_path = "../data/movielens1m.csv"
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train = df_train
    X_test = df_test

    #Skaliranje
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start = datetime.datetime.now()
    model = KMeans(3, 3)
    model.fit(X_train)

    end = datetime.datetime.now() 
    # y_pred = model.predict(X_test)
    # i = 0
    # for p in y_pred:
    #     print(f"Predict {i}: {p}")
    #     i=i+1

    # predicitons = []
    # i = 0
    # for p in X_test:
    #     pred = model.predict(p)
    #     predicitons.append(pred)

    execution_time = end - start

    # score = v_measure_score(y_test,predicitons)
    # print(f"Score: {score}")
    print(f"Execution time: {execution_time.total_seconds()}")

if __name__ == "__main__":
    main()