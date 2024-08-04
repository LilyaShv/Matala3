{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean RMSE: 0.002104743588682707\n",
      "Test RMSE: 0.0022917428597031457\n",
      "Top 5 Features:\n",
      "Price                 0.997719\n",
      "Year                  0.000446\n",
      "Area_עמק יזרעאל       0.000000\n",
      "Area_חולון - בת ים    0.000000\n",
      "Area_חיפה             0.000000\n",
      "dtype: float64\n",
      "Bottom 5 Features:\n",
      "City_גדרה           0.0\n",
      "City_גני תקווה      0.0\n",
      "City_הוד השרון      0.0\n",
      "City_זמר            0.0\n",
      "Color_כחול מטאלי   -0.0\n",
      "dtype: float64\n",
      "ElasticNetCV(cv=10)\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn.linear_model import ElasticNetCV\n",
    "from sklearn.metrics import mean_squared_error, make_scorer\n",
    "import joblib\n",
    "from car_data_prep import prepare_data\n",
    "\n",
    "# קריאת הנתונים\n",
    "df = pd.read_csv('dataset.csv')\n",
    "\n",
    "# הכנת הנתונים\n",
    "X, y = prepare_data(df)\n",
    "\n",
    "# פיצול הנתונים לסט אימון וסט בדיקה\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)\n",
    "\n",
    "# בדיקת גודל המדגם\n",
    "sample_size = min(1000, int(0.8 * len(X_train)))\n",
    "\n",
    "# שימוש במדגם\n",
    "X_sample = X_train.sample(n=sample_size, random_state=42)\n",
    "y_sample = y_train.loc[X_sample.index]\n",
    "\n",
    "# בניית מודל Elastic Net עם cross-validation\n",
    "model = ElasticNetCV(cv=10)\n",
    "\n",
    "# אימון המודל\n",
    "model.fit(X_sample, y_sample)\n",
    "\n",
    "# הערכת הביצועים על ידי RMSE\n",
    "cv_scores = cross_val_score(model, X_sample, y_sample, cv=10, scoring=make_scorer(mean_squared_error, greater_is_better=False))\n",
    "mean_rmse = np.mean((-cv_scores)**0.5)\n",
    "print(f'Mean RMSE: {mean_rmse}')\n",
    "\n",
    "# הערכת הביצועים על סט הבדיקה\n",
    "test_rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)\n",
    "print(f'Test RMSE: {test_rmse}')\n",
    "\n",
    "# זיהוי המאפיינים המשפיעים ביותר\n",
    "feature_importance = pd.Series(model.coef_, index=X_sample.columns).sort_values(ascending=False)\n",
    "top_5_features = feature_importance.head(5)\n",
    "bottom_5_features = feature_importance.tail(5)\n",
    "print(f'Top 5 Features:\\n{top_5_features}')\n",
    "print(f'Bottom 5 Features:\\n{bottom_5_features}')\n",
    "\n",
    "# שמירת המודל המאומן\n",
    "joblib.dump(model, 'trained_model.pkl')\n",
    "\n",
    "import joblib\n",
    "\n",
    "# טוען את המודל המאומן\n",
    "model = joblib.load('trained_model.pkl')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
