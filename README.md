Here I keep track of small projects that I am doing to play with data, learn/pracice algorithms/topics  

## Airline_data_analysis
In  this data visualization project I download flight data from Bureau of Transport Statistics.
- I use pandas for data exploration.
- After cleaning the data I use plotly to make an interactive visualization.
- The result is a plot that shows the aiport locations on a MAP of US, by hovering the mouse on the plot one can get information such as total number of flights, fraction of delayed flights etc.   

## Bus_bunching_Translink
In this supervised learning project I use bus data from Translink (Vancouver's public transport company). One major problem in public buses is that sometimes two buses with the same route number end up in the same bus stop altough they start with a time delay between them. We call this as bus bunching. I use data from trips that occured during October 2016. Using this data I try to model if we can predict if bus bunching will happen. I separate the data into train and test set and use various classification algorithms. 

## NLP_LDA 

In this Natural language processing project I use twitter data and try to identify prevalent topics in the tweets directed at Translink. I use two versions of Latent Dirichlet Allocation- Gensim implementation and Mallet implementation. Due to small sentences in twitter data this technique is not as effective as with longer documents but still gives some idea of the underlying topics. 

## classify_cars

In this computer vision project I use the Stanford cars dataset to identify cars usign their images. Details of this image classification problem are in https://ai.stanford.edu/~jkrause/cars/car_dataset.html

- I use Convolutional Neural Networks.
- I use I use transfer learning, where I use pretrained weights from RESNET50. 
- This was too slow in my computer so I trained the model on Google Cloud Platform. (Machine type 4 vCPUs, 15 GB RAM, NVIDIA Tesla T4X1)
