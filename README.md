# OffNote Labs Research Task

Hi. This is the my implementation of the Vision Task for Offnote Labs.

## Overview

The aim of the project was to find all instances of text embedded in a regular video. This is a popular research problem in Computer Vision, commonly known as Optical Character Recognition (OCR). For this task, I used a video describing [Apple's event in October 2020](https://www.youtube.com/watch?v=Gz8vBoEFArA), since it had a healthy amount of text I could extract.

## Project Structure

I used the following pipeline for this task : 
![image](https://user-images.githubusercontent.com/71698670/121188813-1a298180-c887-11eb-89d5-a5cdc182887f.png)

The first step to split the input video into frames. I used OpenCV’s [EAST text detector](https://arxiv.org/abs/1704.03155) to detect the presence of text in an image. The EAST text detector gives us the bounding box (x, y)-coordinates of text Regions of Interest (ROIs).

We extract each of these ROIs and then pass them into Tesseract v4’s LSTM deep learning text recognition algorithm. 
Tesseract is a very popular OCR engine that has an inbuilt LSTM model for text recognition. The output of the LSTM gives us our actual OCR results.
Finally, we draw the OpenCV OCR results on our output frame and display the results.

## Results

Our OCR engine works fairly well. Here is a snapshot from our text detector : 

https://user-images.githubusercontent.com/71698670/121201611-47c7f800-c892-11eb-832d-dcebb797329d.mp4

And here's the corresponding output from our text recognition system :

![image](https://user-images.githubusercontent.com/71698670/121234397-48bc5200-c8b1-11eb-8014-80bcbeeca766.png)

## Running the Code

1. Download the [EAST Text Detection model](https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1) and [Tesseract OCR engine](https://github.com/UB-Mannheim/tesseract/wiki). I found [this post](https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i) to be quite helpful while setting up Tesseract.
2. Run text_detection.py from the command line by specifying the following arguments:
      
        --image : The path to the input image.
        --east : The path to the pre-trained EAST text detector.

Optionally, the following command line arguments may be provided:

      --min-confidence : The minimum probability of a detected text region.
      --width : The width our image will be resized to prior to being passed through the EAST text detector. Our detector requires multiples of 32.
      --height : Same as the width, but for the height. Again, our detector requires multiple of 32 for resized height.
      --padding : The amount of padding to add to each ROI border

I would also recommend checking out the Jupyter Notebook (text_extraction.ipynb) for a more interactive experience.

### A few observations I made while tuning this model :

**Padding :** A small amount of padding works well for most models. However, as we increase the padding, there may be overlapping between different words in our image/video, leading to poor results from our text recognition engine.

**Frames per second:** The video that I used ran at about 30 FPS. I reduced that to 10 FPS, in order to generate quicker results during test time.

**min confidence:** A higher value of min_confidence gives more accurate bounding boxes, however there's a chance that we may miss a few words that had low probability. I found that a value of 0.5-0.6 tends  to work well for our model 

## Concluding Notes :

Our OCR engine is far from perfect, and it does output a few weird words from time to time :

![image](https://user-images.githubusercontent.com/71698670/121229072-258ea400-c8ab-11eb-9b81-033b1de9fe97.png)


This could be due to specifying incorrect ROIs, or an error by the text recognition engine.  

I loved working on this project, and will be looking for ways to improve it in the future. Working with OffNote Labs would be a great learning experience for me, and I look forward to discussing this oppurtunity with you in the near future.

**Warm Regards**

Shaurya Puri
