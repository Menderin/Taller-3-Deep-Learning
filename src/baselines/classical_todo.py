"""Exercise placeholder for the classical PCA baseline."""


class ClassicalBaseline:
    """TODO(alumno): implement E1 with PCA and classical estimators.

    This is the only strategy that is not expected to use PyTorch for the
    estimator itself. It should still use exactly the same split manifest as
    the neural experiments and report gender and age metrics separately.

    Suggested ablations:
    - Number of PCA components.
    - Gender classifier: GaussianNB or LogisticRegression.
    - Age regressor: Ridge, LinearRegression or RandomForestRegressor.
    """

    def __init__(self, n_components: int = 100, **kwargs):
        from sklearn.decomposition import PCA
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        
        self.n_components = n_components
        
        self.gender_pipeline = Pipeline([
            ("pca", PCA(n_components=self.n_components, whiten=True, random_state=42)),
            ("clf", GaussianNB())
        ])
        
        self.age_pipeline = Pipeline([
            ("pca", PCA(n_components=self.n_components, whiten=True, random_state=42)),
            ("reg", Ridge(alpha=1.0))
        ])

    def fit(self, X_train, y_gender_train, y_age_train, **kwargs) -> None:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        max_components = min(self.n_components, X_train_flat.shape[0] - 1, X_train_flat.shape[1])
        if max_components < 1:
            raise ValueError("Se necesitan al menos dos muestras para entrenar PCA.")
        self.gender_pipeline.set_params(pca__n_components=max_components)
        self.age_pipeline.set_params(pca__n_components=max_components)
        self.gender_pipeline.fit(X_train_flat, y_gender_train)
        self.age_pipeline.fit(X_train_flat, y_age_train)

    def predict(self, X_test, **kwargs):
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        gender_preds = self.gender_pipeline.predict(X_test_flat)
        age_preds = self.age_pipeline.predict(X_test_flat)
        return gender_preds, age_preds
