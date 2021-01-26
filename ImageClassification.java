package ai.certifai.training.MiniProject;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class ImageClassification {

    private static int seed = 123;
    private static Random random_seed = new Random(seed);
    private static int nEpoch = 0;
    private static int nClass = 12;
    private static double lr = 1e-3;                    //nak kena tgk balik
    private static int BatchSize = 0;
    private static int width = 90;
    private static int height = 90;
    private static int nChannel = 3;
    private static double training_percent = 0.7;


    private static final String[] FormatImage = BaseImageLoader.ALLOWED_FORMATS;


    public static void main(String[] args) throws IOException {

        File newInputFile = new ClassPathResource("Chess").getFile();      //nama file
        FileSplit newFileSplite = new FileSplit(newInputFile, FormatImage,random_seed);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(random_seed,FormatImage,labelMaker);

        //Split data training and data test type
        //data[0] = training data ,0.7
        //data[1] = test data , 0.3
        InputSplit[] data = newFileSplite.sample(pathFilter,training_percent,1-training_percent);
        InputSplit Train_data = data[0];
        InputSplit Test_data  = data[1];

        ImageRecordReader ImageFileTrain = new ImageRecordReader(height,width,nChannel,labelMaker);
        ImageRecordReader ImageFileTest = new ImageRecordReader(height,width,nChannel,labelMaker);

        DataNormalization DataNormal = new ImagePreProcessingScaler();






    }

}
