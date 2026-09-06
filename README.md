# Crop-Yield-Prediction-using-Machine-Learning-Algorithms
The project involves building a crop yield prediction model using ML. The first step is to collect data on various factors that can affect crop yield, such as weather patterns, soil quality, fertilization, and irrigation. This data can be obtained from weather stations, soil sensors, satellite imagery, and other sources.

Next, the data needs to be preprocessed to remove any outliers or missing values. Feature engineering techniques can also be applied to extract relevant features from the data that can improve the accuracy of the model.

Once the data is preprocessed, a suitable ML algorithm can be chosen for the task. Commonly used algorithms for crop yield prediction include regression, decision trees, and neural networks. The algorithm can be trained using a subset of the data and evaluated using another subset to measure its performance.

The final step is to deploy the model and make predictions on new data. This can be done using a web interface or a mobile app, which can provide farmers with real-time information about crop yields and help them make informed decisions

---

## Attribution

Original project by [ShubhamKJ123](https://github.com/ShubhamKJ123/Crop-Yield-Prediction-using-Machine-Learning-Algorithms). This fork updates the code to run on modern TensorFlow/Keras and adds a reproducible setup.

### Changes in this fork

- Migrated deprecated Keras imports (`keras.layers.core`, `keras.utils.np_utils`) to the `tensorflow.keras` API
- Removed `_make_predict_function()` calls, unsupported since TensorFlow 2.2
- Added a guard so cancelling the test-data file dialog no longer raises `FileNotFoundError`
- Added `run.bat` launcher, `.gitignore`, and VS Code interpreter settings
- Verified on Python 3.11.9 with TensorFlow 2.15.1
