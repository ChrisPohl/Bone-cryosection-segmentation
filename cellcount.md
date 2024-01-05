# Qupath-Script for Tissue Detection and Cell Count 
## Setting Image Type
```groovy
setImageType('BRIGHTFIELD_H_E');
setColorDeconvolutionStains('{"Name" : "H&E default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049 ", "Stain 2" : "Eosin", "Values 2" : "0.2159 0.8012 0.5581 ", "Background" : " 255 255 255 "}');
```

## Clear Image from prior Annotations
```groovy
resetSelection();
```

## Create Annotation from prior trained Classifier (Detecting Tissue Area): Enter the name of your classifier at “ClassifierX” 
```groovy
createAnnotationsFromPixelClassifier("Classifier X", 3000.0, 1000.0, "DELETE_EXISTING", "SELECT_NEW")
```

## Run Cell Detection
```groovy
runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '{"detectionImageBrightfield": "Hematoxylin OD",  "backgroundRadius": 15.0,  "medianRadius": 0.0,  "sigma": 4.0,  "minArea": 50.0,  "maxArea": 200.0,  "threshold": 0.05,  "maxBackground": 1.0,  "watershedPostProcess": false,  "cellExpansion": 5.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');
```
## Export Measurements

