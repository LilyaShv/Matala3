{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model loaded successfully.\n",
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "   WARNING: This is a development server. Do not use it in a production deployment.\n",
      "   Use a production WSGI server instead.\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "import joblib\n",
    "import pandas as pd\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# טוען את המודל המאומן\n",
    "try:\n",
    "    model = joblib.load('trained_model.pkl')\n",
    "    print(\"Model loaded successfully.\")\n",
    "except Exception as e:\n",
    "    print(f\"Error loading model: {e}\")\n",
    "    exit(1)\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    # מקבל את הנתונים מהבקשה\n",
    "    data = request.json\n",
    "\n",
    "    # המרה ל-DataFrame\n",
    "    input_data = pd.DataFrame(data, index=[0])\n",
    "    \n",
    "    try:\n",
    "        # ביצוע תחזיות\n",
    "        prediction = model.predict(input_data)\n",
    "        return jsonify({'predicted_price': prediction[0]})\n",
    "    except Exception as e:\n",
    "        return jsonify({'error': str(e)}), 400\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True, use_reloader=False)\n"
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
