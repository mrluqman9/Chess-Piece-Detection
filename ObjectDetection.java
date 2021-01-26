package ai.certifai.training.MiniProject;

/*
 * This project is develop for fulfill the requirement that being given from Skymind
 * Project Name:
 * Group members:
 * 1)
 * 2)
 * 3)
 */


import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;

import java.awt.event.KeyEvent;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

public class ObjectDetection {
    private  static final Logger log = org.slf4j.LoggerFactory.getLogger(ObjectDetection.class);
    //Setup For Camera
    private static String Camera = "front";
    private static int CameraNum = 1;
    private static Thread thread = null;

    //For model
    private static int nBox = 5;
    private static double[][] priorBoxes = {{1, 3}, {2.5, 6}, {3, 4}, {3.5, 8}, {4, 9}};
    private static int batchSize = 2;
    private static int nEpochs = 15;
    private static double learningRate = 1e-4;
    private static int nClasses = 12;

    private static int seed = 123;
    private static int width = 14;
    private static int heigh = 14;
    private static double threshold = 0.5;
    private static int YoloWid = 416;
    private static int YoloHei = 416;
    private static List<String> labels;

    //private static File myFile = new File(System.getProperty("C:\\TrainingLabs-main\\Chess"), "generated-models/Chess_yolov2.zip");
    private static ComputationGraph model;
    private static final Scalar GREEN = RGB(0, 255.0, 0);
    private static final Scalar BLUE = RGB(0, 0, 255);
    private static final Scalar RED = RGB(255, 0, 0);
    private static final Scalar BLACK = RGB(0, 0, 0);
    private static final Scalar WHITE = RGB(255, 255, 255);
    private static final Scalar YELLOW = RGB(255, 255, 0);
    private static final Scalar SILVER = RGB(192, 192, 192);
    private static final Scalar PURPLE = RGB(128, 0, 128);
    private static final Scalar ORANGE = RGB(255, 165, 0);
    private static final Scalar BROWN = RGB(165, 42, 42);
    private static final Scalar NAVY = RGB(0, 0, 128);
    private static final Scalar INDIGO = RGB(75, 0, 130);
    private static Scalar[] colormap = {GREEN,BLUE,RED,BLACK,WHITE,YELLOW,SILVER,PURPLE,ORANGE,BROWN,NAVY,INDIGO};
    private static String labeltext = null;
    private static Frame frame = null;

    private static double lambdaNoObj = 0.5;
    private static double lambdaCoord = 5.0;


    public static void main(String[] args) throws Exception {

        String Path = Paths.get(new ClassPathResource("Chess").getPath()).toString();

        //Iterator
        ObjectDetectionIterator.setup(Path);
        RecordReaderDataSetIterator trainIter = ObjectDetectionIterator.trainIterator(batchSize);
        RecordReaderDataSetIterator testIter = ObjectDetectionIterator.testIterator(1);
        labels = trainIter.getLabels();

        Nd4j.getRandom().setSeed(seed);
        INDArray priors = Nd4j.create(priorBoxes);
        ZooModel yolo2 = YOLO2.builder().build();
        ComputationGraph pretrained = (ComputationGraph) YOLO2.builder().build().initPretrained();
        FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();
        model = getComputationGraph(pretrained, priors, fineTuneConf);
        System.out.println(model.summary(InputType.convolutional(
                ObjectDetectionIterator.yoloWidth,
                ObjectDetectionIterator.yoloHeight,
                nClasses)));

        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);
        model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

        for (int i = 1; i < nEpochs + 1; i++) {
            trainIter.reset();
            while (trainIter.hasNext()) {
                model.fit(trainIter.next());
            }
        }


        ModelSerializer.writeModel(model, Path, true);
        System.out.println("Model saved.");

        OfflineValidationWithTestDataset(testIter);
        doInference();
    }


    private static ComputationGraph getComputationGraph(ComputationGraph pretrained, INDArray priors, FineTuneConfiguration fineTuneConf) {

        return new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("conv2d_23")
                .removeVertexKeepConnections("outputs")
                .addLayer("conv2d_23",
                        new ConvolutionLayer.Builder(1, 1)
                                .nIn(1024)
                                .nOut(nBox * (5+nClasses))
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "leaky_re_lu_22")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .lambdaNoObj(lambdaNoObj)
                                .lambdaCoord(lambdaCoord)
                                .boundingBoxPriors(priors.castTo(DataType.FLOAT))
                                .build(),
                        "conv2d_23")
                .setOutputs("outputs")
                .build();
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {

        return new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(learningRate).build())
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();
    }


    //    Evaluate visually the performance of the trained object detection model
    private static void OfflineValidationWithTestDataset(RecordReaderDataSetIterator test) throws InterruptedException {
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame canvas = new CanvasFrame("Validate Test Dataset");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
        Mat convertedMat = new Mat();
        Mat convertedMat_big = new Mat();

        while (test.hasNext() && canvas.isVisible()) {
            org.nd4j.linalg.dataset.DataSet ds = test.next();
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results,threshold);
            YoloUtils.nms(objs, 0.5);
            Mat mat = imageLoader.asMat(features);
            mat.convertTo(convertedMat, CV_8U, 255, 0);
            int w = mat.cols() * 2;
            int h = mat.rows() * 2;
            resize(convertedMat, convertedMat_big, new Size(w, h));
            convertedMat_big = drawResults(objs, convertedMat_big, w, h);
            canvas.showImage(converter.convert(convertedMat_big));
            canvas.waitKey();
        }
        canvas.dispose();
    }

    private static void doInference(){

        NativeImageLoader loader = new NativeImageLoader(
                ObjectDetectionIterator.yoloWidth,
                ObjectDetectionIterator.yoloHeight,
                3,
                new ColorConversionTransform(COLOR_BGR2RGB));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

        if (!Camera.equals("front") && !Camera.equals("back")) {
            try {
                throw new Exception("Unknown argument for camera position. Choose between front and back");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        FrameGrabber grabber = null;
        try {
            grabber = FrameGrabber.createDefault(CameraNum);
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        try {
            grabber.start();
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }

        CanvasFrame canvas = new CanvasFrame("Object Detection");
        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();
        canvas.setCanvasSize(w, h);

        while (true) {
            try {
                frame = grabber.grab();
            } catch (FrameGrabber.Exception e) {
                e.printStackTrace();
            }

            //if a thread is null, create new thread
            if (thread == null) {
                thread = new Thread(() ->
                {
                    while (frame != null) {
                        try {
                            Mat rawImage = new Mat();

                            //Flip the camera if opening front camera
                            if (Camera.equals("front")) {
                                Mat inputImage = converter.convert(frame);
                                flip(inputImage, rawImage, 1);
                            } else {
                                rawImage = converter.convert(frame);
                            }

                            Mat resizeImage = new Mat();
                            resize(rawImage, resizeImage, new Size(ObjectDetectionIterator.yoloWidth, ObjectDetectionIterator.yoloHeight));
                            INDArray inputImage = loader.asMatrix(resizeImage);
                            scaler.transform(inputImage);
                            INDArray outputs = model.outputSingle(inputImage);
                            org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
                            List<DetectedObject> objs = yout.getPredictedObjects(outputs, threshold);
                            YoloUtils.nms(objs, 0.4);
                            rawImage = drawResults(objs, rawImage, w, h);
                            canvas.showImage(converter.convert(rawImage));
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                });
                thread.start();
            }

            KeyEvent t = null;
            try {
                t = canvas.waitKey(33);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }
        }
    }

    private static Mat drawResults(List<DetectedObject> objects, Mat mat, int w, int h) {
        for (DetectedObject obj : objects) {
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            String label = labels.get(obj.getPredictedClass());
            int x1 = (int) Math.round(w * xy1[0] / ObjectDetection.width);
            int y1 = (int) Math.round(h * xy1[1] / ObjectDetection.heigh);
            int x2 = (int) Math.round(w * xy2[0] / ObjectDetection.width);
            int y2 = (int) Math.round(h * xy2[1] / ObjectDetection.heigh);
            //Draw bounding box
            rectangle(mat, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()], 2, 0, 0);
            //Display label text
            labeltext = label + " " + String.format("%.2f", obj.getConfidence() * 100) + "%";
            int[] baseline = {0};
            Size textSize = getTextSize(labeltext, FONT_HERSHEY_DUPLEX, 1, 1, baseline);
            rectangle(mat, new Point(x1 + 2, y2 - 2), new Point(x1 + 2 + textSize.get(0), y2 - 2 - textSize.get(1)), colormap[obj.getPredictedClass()], FILLED, 0, 0);
            putText(mat, labeltext, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, RGB(0, 0, 0));
        }
        return mat;
    }

}