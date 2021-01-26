package ai.certifai.training.MiniProject;

import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;


public class ObjectDetectionIterator  {

    private static int seed = 123;
    private static Random random = new Random(seed);
    private static Path trainDatas,testDatas;
    private static FileSplit trainData,testData;
    private static int width =13;
    private static int height =13;
    private static String[] allowdExt = BaseImageLoader.ALLOWED_FORMATS;
    private static int batchSize  = 2;
    private static int numClass = 3;
    private static int nChannel = 3;
    public static int yoloWidth =416;
    public static int yoloHeight =416;

    public static void setup(String inputDir) {

        trainDatas = Paths.get(inputDir,"Train");
        testDatas = Paths.get(inputDir,"Test");

        trainData = new FileSplit(new File(trainDatas.toString()),allowdExt,random);
        testData = new FileSplit(new File(testDatas.toString()),allowdExt,random);
    }

    private static RecordReaderDataSetIterator makeIterator(InputSplit split, Path dir, int batchSize) throws Exception {

        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(yoloHeight, yoloWidth, nChannel,
                height, width, new VocLabelProvider(dir.toString()));
        recordReader.initialize(split);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 1, true);
        iter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        return iter;
    }

    public static RecordReaderDataSetIterator trainIterator (int batchSize) throws Exception{
        return makeIterator(trainData,trainDatas,batchSize);
    }

    public static RecordReaderDataSetIterator testIterator (int batchSize) throws Exception{
        return makeIterator(testData,testDatas,batchSize);
    }
}