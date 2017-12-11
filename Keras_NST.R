# Neural style transfer in Keras
# 
# Neural style transfer can be implemented using any pre-trained convnet. 
# Here we will use the VGG19 network, used by Gatys et al in their paper. 
# VGG19 is a simple variant of the VGG16 network with three more convolutional 
# layers.
# 
# This is our general process:
#   
# 1. Set up a network that will compute VGG19 layer activations for the style 
#    reference image, the target image, and the generated image at the same time.
# 2. Use the layer activations computed over these three images to define the 
#    loss function described above, which we will minimize in order to achieve 
#    style transfer.
# 3. Set up a gradient descent process to minimize this loss function.
#
# Let's start by defining the paths to the two images we consider: the style 
# reference image and the target image. To make sure that all images processed 
# share similar sizes (widely different sizes would make style transfer more 
# difficult), we will later resize them all to a shared height of 400px.
library(keras)
K <- backend()

# This is the path to the image you want to transform.
target_image_path <- file.path(".", "content_images", "IMG_1213.JPG")

# This is the path to the style image.
style_reference_image_path <- file.path(".", "style_images", "Starry_Night.jpg")

# Dimensions of the generated picture.
img <- image_load(target_image_path)
width <- img$size[[1]]
height <- img$size[[2]]
img_nrows <- 400
img_ncols <- as.integer(width * img_nrows / height)  

# We will need some auxiliary functions for loading, pre-processing and 
# post-processing the images that will go in and out of the VGG19 convnet:

preprocess_image <- function(path) {
  img <- image_load(path, target_size = c(img_nrows, img_ncols)) %>%
    image_to_array() %>%
    array_reshape(c(1, dim(.)))
  imagenet_preprocess_input(img)
}

deprocess_image <- function(x) {
  x <- x[1,,,]
  # Remove zero-center by mean pixel
  x[,,1] <- x[,,1] + 103.939
  x[,,2] <- x[,,2] + 116.779
  x[,,3] <- x[,,3] + 123.68
  # 'BGR'->'RGB'
  x <- x[,,c(3,2,1)]
  x[x > 255] <- 255
  x[x < 0] <- 0
  x[] <- as.integer(x)/255
  x
}

# Let's set up the VGG19 network. It takes as input a batch of three images: 
# the style-reference image, the target image, and a placeholder that will 
# contain the generated image. 
#
# A placeholder is a symbolic tensor, the values of which are provided 
# externally via R arrays. The style-reference and target 
# image are static and thus defined using K$constant, whereas the values 
# contained in the placeholder of the generated image will change over time.

target_image <- K$constant(preprocess_image(target_image_path))
style_reference_image <- K$constant(
  preprocess_image(style_reference_image_path)
)

# This placeholder will contain our generated image
combination_image <- K$placeholder(c(1L, img_nrows, img_ncols, 3L)) 

# We combine the 3 images into a single batch
input_tensor <- K$concatenate(list(target_image, style_reference_image, 
                                   combination_image), axis = 0L)

# We build the VGG19 network with our batch of 3 images as input.
# The model will be loaded with pre-trained ImageNet weights.
model <- application_vgg19(input_tensor = input_tensor, 
                           weights = "imagenet", 
                           include_top = FALSE)

cat("Model loaded\n")

# CONTENT LOSS
#
# Let's define the content loss, meant to make sure that the top layer of 
# the VGG19 convnet will have a similar view of the target image and the 
# generated image:
content_loss <- function(base, combination) {
  K$sum(K$square(combination - base))
}

# STYLE LOSS
#
# Now, here's the style loss. It leverages an auxiliary function to compute 
# the Gram matrix of an input matrix, i.e. a map of the correlations found in 
# the original feature matrix.
gram_matrix <- function(x) {
  features <- K$batch_flatten(K$permute_dimensions(x, list(2L, 0L, 1L)))
  gram <- K$dot(features, K$transpose(features))
  gram
}

style_loss <- function(style, combination){
  S <- gram_matrix(style)
  C <- gram_matrix(combination)
  channels <- 3
  size <- img_nrows*img_ncols
  K$sum(K$square(S - C)) / (4 * channels^2  * size^2)
}

# TOTAL VARIATION LOSS
#
# To these two loss components, we add a third one, the "total variation loss". 
# It is meant to encourage spatial continuity in the generated image, thus 
# avoiding overly pixelated results. You could interpret it as a 
# regularization loss.

total_variation_loss <- function(x) {
  y_ij  <- x[,1:(img_nrows - 1L), 1:(img_ncols - 1L),]
  y_i1j <- x[,2:(img_nrows), 1:(img_ncols - 1L),]
  y_ij1 <- x[,1:(img_nrows - 1L), 2:(img_ncols),]
  a <- K$square(y_ij - y_i1j)
  b <- K$square(y_ij - y_ij1)
  K$sum(K$pow(a + b, 1.25))
}

# The loss that we minimize is a weighted average of these three losses. 
# To compute the content loss, we only leverage one top layer, 
# the block5_conv2 layer, while for the style loss we use a list of layers 
# than spans both low-level and high-level layers. We add the total variation 
# loss at the end.
# 
# Depending on the style reference image and content image you are using, you 
# will likely want to tune the content_weight coefficient, the contribution of 
# the content loss to the total loss. A higher content_weight means that the 
# target content will be more recognizable in the generated image.
# Named list mapping layer names to activation tensors
outputs_dict <- lapply(model$layers, `[[`, "output")
names(outputs_dict) <- lapply(model$layers, `[[`, "name")

# Name of layer used for content loss
content_layer <- "block5_conv2" 

# Name of layers used for style loss
style_layers = c("block1_conv1", "block2_conv1",
                 "block3_conv1", "block4_conv1",
                 "block5_conv1")

# Weights in the weighted average of the loss components
total_variation_weight <- 1e-4
style_weight <- 1.0
content_weight <- 0.025

# Define the loss by adding all components to a `loss` variable
loss <- K$variable(0.0) 
layer_features <- outputs_dict[[content_layer]] 
target_image_features <- layer_features[1,,,]
combination_features <- layer_features[3,,,]

loss <- loss + content_weight * content_loss(target_image_features,
                                             combination_features)

for (layer_name in style_layers){
  layer_features <- outputs_dict[[layer_name]]
  style_reference_features <- layer_features[2,,,]
  combination_features <- layer_features[3,,,]
  sl <- style_loss(style_reference_features, combination_features)
  loss <- loss + ((style_weight / length(style_layers)) * sl)
}

loss <- loss + 
  (total_variation_weight * total_variation_loss(combination_image))

# Finally, we set up the gradient-descent process. In the original Gatys et al. 
# paper, optimization is performed using the L-BFGS algorithm, so that is also 
# what you'll use here. The L-BFGS algorithm is available via the optim() 
# function, but there are two slight limitations with the optim() 
# implementation:
# 
# - It requires that you pass the value of the loss function and the value of 
#   the gradients as two separate functions.
# - It can only be applied to flat vectors, whereas you have a 3D image array.
#
# It would be inefficient to compute the value of the loss function and the 
# value of the gradients independently, because doing so would lead to a lot 
# of redundant computation between the two; the process would be almost twice 
# as slow as computing them jointly. To bypass this, you'll set up an R6 class 
# named Evaluator that computes both the loss value and the gradients value at 
# once, returns the loss value when called the first time, and caches the 
# gradients for the next call.

# Get the gradients of the generated image wrt the loss
grads <- K$gradients(loss, combination_image)[[1]] 

# Function to fetch the values of the current loss and the current gradients
fetch_loss_and_grads <- K$`function`(list(combination_image), list(loss, grads))

eval_loss_and_grads <- function(image) {
  image <- array_reshape(image, c(1, img_nrows, img_ncols, 3))
  outs <- fetch_loss_and_grads(list(image))
  list(
    loss_value = outs[[1]],
    grad_values = array_reshape(outs[[2]], dim = length(outs[[2]]))
  )
}

library(R6)
Evaluator <- R6Class("Evaluator",
                     public = list(
                       
                       loss_value = NULL,
                       grad_values = NULL,
                       
                       initialize = function() {
                         self$loss_value <- NULL
                         self$grad_values <- NULL
                       },
                       
                       loss = function(x){
                         loss_and_grad <- eval_loss_and_grads(x)
                         self$loss_value <- loss_and_grad$loss_value
                         self$grad_values <- loss_and_grad$grad_values
                         self$loss_value
                       },
                       
                       grads = function(x){
                         grad_values <- self$grad_values
                         self$loss_value <- NULL
                         self$grad_values <- NULL
                         grad_values
                       }
                     )
)

evaluator <- Evaluator$new()

# Finally, you can run the gradient-ascent process using the L-BFGS algorithm, 
# plotting the current generated image at each iteration of the algorithm 
# (here, a single iteration represents 20 steps of gradient ascent).
iterations <- 20

dms <- c(1, img_nrows, img_ncols, 3)

# This is the initial state: the target image.
x <- preprocess_image(target_image_path)
# Note that optim can only process flat vectors.
x <- array_reshape(x, dim = length(x))  

for (i in 1:iterations) { 
  cat("ITERATION ", i, "\n")
  
  # Runs L-BFGS over the pixels of the generated image to minimize the neural style loss.
  opt <- optim(
    array_reshape(x, dim = length(x)), 
    fn = evaluator$loss, 
    gr = evaluator$grads, 
    method = "L-BFGS-B",
    control = list(maxit = 15)
  )
  
  cat("Loss:", opt$value, "\n")
  
  image <- x <- opt$par
  image <- array_reshape(image, dms)
  
  im <- deprocess_image(image)
  plot(as.raster(im))
  
  image_array_save(im, file.path(".", "results", sprintf("output_%d.jpg", i)))
}