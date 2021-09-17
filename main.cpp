/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <opencv2/opencv.hpp>
#include <vector>

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

/*
void readImageForTF(std::unique_ptr<tflite::FlatBufferModel> model) {
  // Input and output Tensors
//  tflite::Tensor input; //(*model.get(), "serving_default_input_layer");
//  tflite::Tensor prediction(model, "StatefulPartitionedCall");

  // Read image and convert to float
  cv::Mat image = cv::imread("intel-dataset/8.jpg");
  image.convertTo(image, CV_32F, 1.0/255.0);

// Image dimensionsT* output = interpreter->typed_output_tensor<T>(i);
  auto IMG_SIZE = 192;
  int rows = image.rows;
  int cols = image.cols;
  int channels = image.channels();
  int total = image.total();

// Assign to vector for 3 channel image
// Souce: https://stackoverflow.com/a/56600115/2076973
  cv::Mat flat = image.reshape(1, image.total() * channels);

  std::vector<float> img_data(IMG_SIZE*IMG_SIZE*3);
  img_data = image.isContinuous()? flat : flat.clone();

  // Feed data to input tensor
  model->
  input.set_data(img_data, {1, rows, cols, channels});

// Run and show predictions
  model->run(input, prediction);

// Get tensor with predictions
  std::vector<float> predictions = prediction.Tensor::get_data<float>();
  for(int i=0; i<predictions.size(); i++)
    std::cout<< std::to_string(predictions[i]) << std::endl;
}
*/

cv::Mat getSquareImage( const cv::Mat& img, int target_width = 500 )
{
  int width = img.cols,
      height = img.rows;

  cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

  int max_dim = ( width >= height ) ? width : height;
  float scale = ( ( float ) target_width ) / max_dim;
  cv::Rect roi;
  if ( width >= height )
  {
    roi.width = target_width;
    roi.x = 0;
    roi.height = height * scale;
    roi.y = ( target_width - roi.height ) / 2;
  }
  else
  {
    roi.y = 0;
    roi.height = target_width;
    roi.width = width * scale;
    roi.x = ( target_width - roi.width ) / 2;
  }

  cv::resize( img, square( roi ), roi.size() );

  return square;
}

void printMinMaxValues(cv::Mat &rgbImage) {
  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;
  std::vector<cv::Mat> channelsVec(3);
  // split img:
  split(rgbImage, channelsVec);
  cv::minMaxLoc( channelsVec.at(1), &minVal, &maxVal, &minLoc, &maxLoc);
  std::cout << "min val: " << minVal << std::endl;
  std::cout << "max val: " << maxVal << std::endl;
}

std::string getStringFromDepth(int depthVal) {
  switch (depthVal) {
    case 0: return "CV_8U ";
    case 1: return "CV_8S ";
    case 2: return "CV_16U";
    case 3: return "CV_16S";
    case 4: return "CV_32S";
    case 5: return "CV_32F";
    case 6: return "CV_64F";
    case 7: return "CV_16F";
    default: return "UNKNOWN";
  }
}

void printImageDetails(cv::Mat &img) {
  std::cout << "\nheight: " << img.rows << std::endl;
  std::cout << "width : " << img.cols << std::endl;
  std::cout << "channl: " << img.channels() << std::endl;
  std::cout << "depth : " << getStringFromDepth(img.depth()) << std::endl;
  std::cout << "total : " << img.total() << std::endl;
}




#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  auto IMG_SIZE = 192;
  auto image = cv::imread("/Users/luca/Develop/tensorflowTest/lo.jpeg", cv::IMREAD_COLOR);
  auto squaredImage = getSquareImage(image, IMG_SIZE);
  auto inputImage = getSquareImage(image, IMG_SIZE);
  cv::Mat rgbImage;
  cv::cvtColor(inputImage, rgbImage, cv::COLOR_BGR2RGB);
  rgbImage.convertTo(rgbImage, CV_32FC3);
  printMinMaxValues(rgbImage);
  printImageDetails(rgbImage);
  // Assign to vector for 3 channel image
  cv::Mat flat = rgbImage.reshape(1, rgbImage.total() * rgbImage.channels());
  std::vector<float> img_data(IMG_SIZE*IMG_SIZE*3);
  img_data = rgbImage.isContinuous()? flat : flat.clone();
  printImageDetails(flat);
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`


//  inputPtr = img_data.data();
//  int* outtensor = interpreter->typed_tensor<int>(0);
//  auto input_type = interpreter->tensor(0)->type;
//  interpreter->SetInputs(img_data);
  // Converting to vector
//  std::vector<uchar> vec(flat.data, flat.data + flat.total());
  std::cout << "Inputs.size: " << interpreter->inputs().size() << std::endl;
  std::cout << "Input name 0: " << interpreter->GetInputName(0) << std::endl;
  auto* inputPtr = interpreter->typed_input_tensor<float>(0);
  auto* ptr2 = interpreter->input_tensor(0);
  for (size_t i = 0; i < flat.total(); ++i) {
    inputPtr[i] = img_data[i];
//    interpreter->typed_input_tensor<uchar>(0)[3*i + 1] = rgb[1];
//    interpreter->typed_input_tensor<uchar>(0)[3*i + 2] = rgb[2];
  }

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
  // output is a float32 tensor of shape [1, 1, 17, 3]
  // 17: [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]
  //  3: yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) and prediction confidence scores of each keypoint, also in the range [0.0, 1.0]
  auto* output = interpreter->typed_output_tensor<float>(0);
  std::vector<std::vector<float>> out;
  for (int i = 0; i < 17; ++i) {
    out.emplace_back(std::vector<float>({output[i*3 + 0], output[i*3 + 1], output[i*3 + 2]}));
  }
//  auto pOut = out.data();
//  pOut = output;

  // Testing by reconstruction of cvMat
//  cv::Mat restored = cv::Mat(inputImage.rows, inputImage.cols, inputImage.type(), input); // OR vec.data() instead of ptr
  for (auto val : out) {
    std::cout << "x: " << val.at(0) << "| y: " << val.at(1) << "| p: " << val.at(2) << std::endl;
    auto center = cv::Point(int(val.at(1) * IMG_SIZE), int(val.at(0) * IMG_SIZE));
    circle(squaredImage, center,1, CV_RGB(255,0,0),3);
  }
  cv::namedWindow("reconstructed", cv::WINDOW_AUTOSIZE);
  cv::imshow("reconstructed", squaredImage);
  cv::waitKey(0);
  return 0;
}
