from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import joblib


def train_model():

    # prepocess
    X, y = datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # train
    model = SGDRegressor(verbose=1)
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')

    # test
    y_pred = model.predict(X_test)
    err = mean_squared_error(y_test, y_pred)
    print(err)

    # deploy
    print('deploying model...')


if __name__ == '__main__':
    train_model()
