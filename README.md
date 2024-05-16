# Leaf Identification Using Convolutional Neural Networks

Hey! I'm Balaji! (pronounced Baa Laa Jee)
I'm creating this page to highlight what I've learnt in my Final Year Project in my course "Computer Science Engineering" at "Jain University, Bangalore " in 2022-2023.

The primary goal of this project was towards modelling a system using deep learning techniques like convolutional neural networks and creating a dataset of our own requirements featuring "Local Indian plant leaves" .

Even for the expert botanists, species identification is often a laborious task. Manual identification is often time-consuming and inefficient.

For non-experts who lack botanical training and are unfamiliar with the terms used in the field, Identification of plants becomes difficult and tough even if professionals have the knowledge and expertise necessary to identify plant species. Recognizing this difficulty, scientists and researchers are investigating novel strategies to aid in species identification, including the use of deep learning methods like Convolutional Neural Networks (CNNs).
By automatically recognizing complex patterns and characteristics from leaf photos, CNNs provide a potential method that enables automated and accurate species identification.

We created this machine learning model in roughly 12 months. We also created the dataset by ourselves using images shot on smartphones. These images were then scaled down and processed to create the dataset to train the Machine Learning model in our project.


# So what are Convolutional Neural Networks?
A deep learning algorithm known as a convolutional neural network (CNN) was developed upfront for
handling and analyzing visual input, such as pictures. It is a type of artificial neural network that takes
its structure and operation ideas from the human brain. CNNs are quite effective in applications like
image classification, object identification, and image segmentation.


The capacity of CNNs to automatically learn and extract useful characteristics from unprocessed input
data is its distinguishing characteristic. Convolutional layers, which apply localized operations on the
input data, are used to achieve this. The CNN is able to identify significant patterns and characteristics,
including edges, textures, and forms, by applying filters to discrete areas of the input.
Convolutional, pooling, and fully linked layers are among the layers that make up a standard CNN's
architecture. Pooling layers down sample the derived features to minimize computational complexity
and enhance translation invariance after convolutional layers extract features from the input data. The
classification or regression job is carried out by fully connected layers at the network's end using the
learnt features.


Backpropagation is a technique used by CNNs to optimize its internal parameters during training. In this
technique, the network modifies the weights of its neurons to reduce the discrepancy between anticipated
outputs and true labels. CNNs can enhance their performance over time by progressively learning to
recognize and distinguish between various visual patterns thanks to this iterative learning process.


CNNs have revolutionized computer vision and delivered excellent results in a variety of applications.
They are extremely adaptive and capable of performing difficult visual tasks because to their capacity to
automatically learn and extract characteristics from pictures without the need for manual feature
engineering. CNNs can learn to discriminate between diverse leaf traits and properly categorizes leaves
into their respective plant species in the context of plant leaf species identification.

# Why CNNs? What are its advantages?

Convolutional Neural Networks (CNNs) have numerous benefits that make them ideal for a variety of
computer vision tasks. Here are some important benefits of CNNs:
• CNNs are capable of automatically learning and extracting useful characteristics from raw input
data. CNNs may find and capture important patterns and structures in data by using convolutional
layers and pooling layers. Because the network learns to extract important features on its own,
this feature learning capacity eliminates the need for human feature engineering.
• CNNs have translation invariance, which means they can recognize patterns and features
regardless of where they are in an image. This trait is achieved by employing pooling layers,
which down sample feature maps and capture the most important information. Translation
invariance allows CNNs to recognize objects or features even when they are shifted or modified
in the input data.
• CNNs use parameter sharing to lower the number of parameters and enable the network to learn
from fewer data samples. The same set of weights (filter/kernel) is applied across multiple spatial
regions of the input data in a convolutional layer. This parameter sharing considerably decreases
the network's memory footprint and facilitates quick training, especially when working with large
datasets.
• CNNs learn a hierarchical representation of the incoming data on their own. Deeper layers
eventually acquire more abstract and high-level representations while the earliest layers capture
low-level elements such as edges and textures. This hierarchical structure allows the network to
comprehend complicated connections and meanings in the data, allowing it to recognize objects and 
make fine-grained distinctions between various classes.
Spatial Hierarchy Capture: CNNs excel at capturing spatial hierarchies inside pictures. Local
receptive fields are processed by the convolutional layers, allowing the network to learn spatial
correlations between neighbouring pixels or areas. Understanding the spatial hierarchy is critical
for tasks like object identification, segmentation, and localization.
• Beyond pictures: While CNNs are commonly utilized for image-related tasks, their use goes
beyond pictures. CNN architectures have been successfully extended and used to additional data
modalities including as text, time series, and audio. This flexibility demonstrates CNNs'
adaptability and capability in gathering and extracting information from many forms of data.
• Continuous Learning: CNNs may be trained incrementally or adaptively, letting them to learn
from new data without forgetting what they learnt from previous data. CNNs can manage
dynamic and developing datasets thanks to continuous learning techniques, making them suited
for circumstances where the data distribution changes over time.
• CNNs are resistant to changes in input data, such as changes in size, rotation, or occlusion. The
hierarchical feature learning process, in which lower-level characteristics are merged to produce
higher-level representations, achieves this resilience. As a result, CNNs can deal with fluctuations
in object appearance, allowing them to be useful in real-world settings with diverse and difficult
conditions.

# What are our objectives?

Custom datasets are essential for training CNNs for specific applications because they allow
researchers and practitioners to customize the data to their specific needs. They have control over
the data quality, annotation procedure, and dataset size by developing their own datasets, enabling
for more targeted and accurate training of CNN models. Custom datasets are essential for
training CNNs for specific applications because they allow researchers and practitioners
to customize the data to their specific needs. They have control over the data quality,
annotation procedure, and dataset size by developing their own datasets, enabling for
more targeted and accurate training of CNN models.
• In our model we are using custom dataset because of the following advantages:
Custom datasets are designed to fulfil the particular requirements and objectives of a
project. They are intended to collect the required data and characteristics to train a CNN
for a given job, such as image classification, object identification, or semantic
segmentation. The dataset's scope is specified by the precise classes or categories of
interest as well as the task's difficulty.
• Data Collection: Depending on the nature of the project, custom datasets can be acquired
in a variety of ways. Capturing photos or videos using cameras or sensors, obtaining text
data from specified sources, or collecting data from specialized equipment or instruments
are all examples of data collection. The gathering procedure may involve manual efforts
like as annotation or labelling, or the use of automated tools.
• Labelling and annotation: Annotation and labelling are frequently required for custom
datasets, which entails manually or automatically giving class labels or annotations to the
data. This stage is critical for supervised learning, which involves training CNNs using
labelled samples. Bounding boxes for object recognition, pixel-level masks for
segmentation, and category labels for classification are examples of annotations. The
annotation procedure guarantees that the CNN learns to link the appropriate labels with
the appropriate data samples.
The amount of a bespoke dataset might vary based on the project needs and the task's
complexity. Custom datasets can range in size from a few hundred to a few thousand
samples to big datasets containing millions of samples. The dataset size should be large
enough to give adequate variety.
• Data Stabilization: To achieve equal representation across multiple classes or categories,
custom datasets may need careful analysis of class distribution. Imbalanced datasets, in
which some classes contain disproportionately more samples than others, can bias CNN
learning and lead to poor performance in underrepresented classes. To overcome class
imbalance difficulties, balancing approaches such as oversampling, under sampling, or
data augmentation may be used.
• Data Preprocessing: Custom datasets are frequently subjected to preprocessing
techniques in order to improve the data's quality and usefulness. Preprocessing may
include picture scaling or cropping to a consistent size, normalizing pixel values to a given
range, eliminating noise or artefacts, and dealing with missing or incorrect data. Data
preparation ensures that the input data is in the correct format for the CNN and aids in its
performance.
Custom datasets are essential for training CNNs for specific applications because they allow
researchers and practitioners to customize the data to their specific needs. They have control over
the data quality, annotation procedure, and dataset size by developing their own datasets, enabling
for more targeted and accurate training of CNN models.

# What's our methodology?

The project was managed through three sections of steps. These steps are 
1)Dataset Creation
2)Neural network modeling 
3)Implementation & Testing

Each of these sections are a multi-step process. We will talk
about each section of these steps in detail.

# 1. Dataset
This is the first section of the project dealing with generation of dataset and relevant research to be done.
We have summarized all the steps into three important steps. They are Acquisition of Images for the
Dataset, Applying Data augmentation techniques to generate more images, and finally processing the
images into machine understandable data and encoding the data with respective classes.

a. Image Acquisition:
  • The process of capturing or getting leaf images that are utilized as input data for convolutional
  neural network (CNN) training and testing is known as image acquisition in leaf identification
  using CNN. To do this, high-quality leaf photos must be captured using a variety of tools,
  including digital cameras, scanners, and even mobile devices. After the obtained images have
  been preprocessed, the CNN model is used to further analyses and classify them.
  
b. Data Augmentation:
  • Data augmentation strategies can be used to expand the dataset and enhance the model's
  stability.
  • By using changes like rotation, zooming, scaling, and flipping, data augmentation creates
  new versions of the current leaf images.

  ![image](https://github.com/bxlxji/Final-Year-Project-BTech-2022-2023/assets/79824566/70c41164-7523-49eb-9472-9110a76d8bd2)

c. Processing And Encoding
  • Preprocessing: To improve the quality and consistency of the leaf images, different
  preprocessing techniques are utilized in this step. Resizing the pictures to a standard size,
  clipping the background out of the images that aren't relevant, and using filters to get removal
  of noise or glitches are all common preprocessing techniques.
  • Dataset Splitting: Splitting a dataset into three groups for training, validation, and testing is
  the norm. The validation set is used to fine-tune the model and make decisions about
  hyperparameters, the testing set is used to assess the trained model's ultimate performance,
  and the training set is used to train the CNN model.
  • Labeling: Each image of a leaf in the dataset has a corresponding label or class label that
  identifies the species or group of leaves. The labels are applied to the photographs either
  manually by specialists or automatically via techniques.
  • Encoding: Labels are encoded in a format that will be useful for training the CNN model.
  Usually, this entails translating the category labels into numerical forms. One-hot encoding
  is a popular encoding method where each label

# 2. Convolutional Neural Network
This section constitutes designing the model fit for the requirement and can be able to handle dataset.
Involving three step process of Feature Extraction, Modeling the convolutional network and compiling
and training the model.

![image](https://github.com/bxlxji/Final-Year-Project-BTech-2022-2023/assets/79824566/9a7004f2-7685-48a9-ba09-303fa161220a)

a. Feature Extraction:
  • In order to train the convolutional neural network (CNN) model for leaf identification,
  informative and discriminative features must be extracted from the leaf images. The model
  gains an understanding of the distinctive traits and patterns that set one leaf species apart from
  another through this process.
  • The CNN model can discriminate between several leaf species based on their distinctive
  visual properties by extracting information through the convolutional layers and learning
  discriminative representations through the fully connected layers.
  
b. Layering (Modeling):
  • During the modeling process, the number and size of the convolutional and pooling layers,
  as well as the number of fully connected layers, can be adjusted based on the complexity of
  the leaf identification task and the available computational resources. The CNN model is
  trained using labeled leaf images, and the weights and biases of the network are optimized to
  minimize the classification error. Once trained, the model can be used to classify new leaf
  images accurately.
  
c. Compiling And Training The Model:
Compiling and training the model involves configuring the model for training and fitting it to the
labeled leaf images. Here's a concise summary:
  • Compilation: Configure the model with a loss function, optimizer, and evaluation metrics.
  • Training: Fit the model to the labeled leaf images, adjusting its weights to minimize the loss
  function through multiple epochs.
  • Validation: Monitor the model's performance on a separate validation dataset during training
  to prevent overfitting.
  • Evaluation: Assess the trained model's performance on a separate testing dataset using
  relevant metrics.
By compiling and training the model, it learns to classify and identify leaf species based on the
extracted features from the images.

# 3. Testing And Implementation

a. Testing The Model:
To test the model for leaf identification using CNN, follow these steps:
  • Prepare a separate dataset of labeled leaf images that were not used during training or
  validation.
  • Input the unseen leaf images into the trained CNN model.
  • Obtain the predicted class labels for the leaf images from the model's output.
  • Compare the predicted labels with the true labels of the leaf images.
  • Calculate evaluation metrics such as accuracy, precision, recall, or F1-score to assess the
  model's performance.
  • Analyze the model's performance on the testing dataset to determine its accuracy in correctly
  identifying the leaf species.
By testing the model on unseen leaf images, you can evaluate its ability to generalize and make
accurate predictions on new data.

b. Reworking The Model:
To rework the model for leaf identification using CNN, consider the following steps:
  • Evaluate the current model's performance and identify areas for improvement, such as
  accuracy or robustness.
  • Experiment with different model architectures by adjusting the number and size of
  convolutional layers, pooling layers, and fully connected layers.
  • Try different activation functions, such as ReLU or Leaky ReLU, to introduce non-linearity
  and enhance feature extraction.
  • Incorporate regularization techniques like dropout or batch normalization to reduce
  overfitting and improve generalization.
  • Explore transfer learning by using pre-trained CNN models, such as VGG or ResNet, as a
  starting point and fine-tuning them on your leaf dataset.
  • Consider increasing the size of the training dataset or applying data augmentation techniques
  to improve model performance.
  • Optimize hyperparameters, such as learning rate, batch size, and optimizer, to find the best
  configuration for your leaf identification task.
  • Regularly evaluate the reworked model's performance on validation and testing datasets to
  ensure progress and make adjustments as needed.
  By reworking the model architecture and experimenting with different components and techniques,
  you can enhance the model's performance in leaf identification using CNN.
  
c. Implementing End User Interface:
To implement an end-user interface for leaf identification using CNN, follow these steps:
  • Design the User Interface (UI): Create an intuitive and user-friendly UI where users can
  interact with the leaf identification system. Consider including features like image upload,
  result display area, and any additional functionalities you want to provide.
  • Image Upload: Allow users to upload leaf images either from their device or through a
  camera interface. Ensure that the image format and size requirements are specified.
  • Preprocess Images: Preprocess the uploaded images to ensure they are in the appropriate
  format and size for the CNN model. Resize, crop, or normalize the images as necessary.

Model Inference: Pass the preprocessed leaf images through the trained CNN model to
obtain predictions or probabilities for each leaf class.
  • Display Results: Present the results to the user, showing the predicted leaf species and any
  additional information you want to provide, such as confidence scores or top-k predictions.
  • Error Handling: Implement error handling mechanisms to handle cases where the uploaded
  image is not valid or the model encounters any issues during inference.
  • User Feedback: Consider including a feedback mechanism to collect user feedback on the
  system's performance or allow users to report misclassifications.
  • Continuous Improvement: Regularly update and improve the model based on user feedback
  and additional training data to enhance the system's accuracy and usability.
By implementing an end-user interface, users can easily interact with the leaf identification system,
upload images for classification, and view the results, providing a user-friendly experience for
identifying leaf species using CNN.

