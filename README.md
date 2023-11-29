# Lettuce-freshweight-dryweight-leafarea-predition-DL-EfficientNetB6
A simple implementation for a course project. The topic is to predict the fresh weight, dry weight, and leaf area with EfficientNet B7 to advance production in controlled environments such as greenhouses and vertical farms.  
The image dataset and ground truth download link is here: https://library.wur.nl/WebQuery/wurpubs/586469  Be advised, the RGB_277/309/322.png are missing, so are Depth_277/309/322.png
To run this code, you need:
1. The source code (project.py)
2. The image dataset: Downloaded link: https://library.wur.nl/WebQuery/wurpubs/586469
3. The trained model(Optional): Download link https://www.dropbox.com/scl/fi/pbqemgbq8zg1mn92ptn7x/best_model_weights-b6.pth?rlkey=2bdtbt4zerau9nanr1esxpvg4&dl=0
![RGB_140](https://github.com/Kaiwen-Xiao/Lettuce-freshweight-dryweight-leafarea-predition-DL-EfficientNetB6/assets/126135993/1951f313-b7d4-4074-9e30-639fca3794f2)

Trained for about 1000 epochs.

Total Validation Loss: 87532.714

Total Train Loss: 44181.341
![image](https://github.com/Kaiwen-Xiao/Lettuce-freshweight-dryweight-leafarea-predition-DL-EfficientNetB6/assets/126135993/2e73315e-dfbe-4afa-bb29-6a05fa8d426a)

RMSE:
Fresh weight: 32.91(Val); 40.96 (Train)
Dry weight: 1.21 (Val); 1.41 (Train)
Leaf area: 511.38 (Val); 361.75 (Train)
R²:
Fresh weight: 0.916 (Val); 0.857 (Train)
Dry weight: 0.930 (Val); 0.903 (Train)
Leaf area: 0.918 (Val); 0.941 (Train)
![image](https://github.com/Kaiwen-Xiao/Lettuce-freshweight-dryweight-leafarea-predition-DL-EfficientNetB6/assets/126135993/f8a0c229-4440-44c1-bc45-1ae1e1c08618)

   

    


