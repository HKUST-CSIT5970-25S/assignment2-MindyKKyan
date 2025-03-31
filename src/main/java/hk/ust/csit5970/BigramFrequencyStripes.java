package hk.ust.csit5970;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

/**
 * Hadoop MapReduce implementation to calculate bigram frequencies using stripe approach.
 * Input: Text documents
 * Output: Relative frequencies of bigrams (word pairs)
 */
public class BigramFrequencyStripes extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(BigramFrequencyStripes.class);

    /**
     * Mapper class that generates stripes of bigrams.
     * Input: LongWritable (offset), Text (line of document)
     * Output: Text (first word), HashMapStringIntWritable (stripe of second words with counts)
     */
    private static class MyMapper extends Mapper<LongWritable, Text, Text, HashMapStringIntWritable> {
        private static final Text KEY = new Text();
        private static final HashMapStringIntWritable STRIPE = new HashMapStringIntWritable();

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            String[] words = line.trim().split("\\s+");
            
            // Generate bigrams and create stripes
            for (int i = 0; i < words.length - 1; i++) {
                String w1 = words[i];
                String w2 = words[i + 1];
                
                // Emit (w1, {w2:1}) pair
                KEY.set(w1);
                STRIPE.clear();
                STRIPE.put(w2, 1);
                context.write(KEY, STRIPE);
            }
        }
    }

    /**
     * Reducer class that aggregates stripes and calculates relative frequencies.
     * Input: Text (first word), Iterable<HashMapStringIntWritable> (stripes)
     * Output: PairOfStrings (bigram), FloatWritable (relative frequency)
     */
    private static class MyReducer extends Reducer<Text, HashMapStringIntWritable, PairOfStrings, FloatWritable> {
        private final static HashMapStringIntWritable SUM_STRIPES = new HashMapStringIntWritable();
        private final static PairOfStrings BIGRAM = new PairOfStrings();
        private final static FloatWritable FREQ = new FloatWritable();

        @Override
        public void reduce(Text key, Iterable<HashMapStringIntWritable> stripes, Context context)
                throws IOException, InterruptedException {
            SUM_STRIPES.clear();
            
            // Aggregate all stripes for current key
            for (HashMapStringIntWritable stripe : stripes) {
                for (Map.Entry<String, Integer> entry : stripe.entrySet()) {
                    SUM_STRIPES.increment(entry.getKey(), entry.getValue());
                }
            }

            // Calculate total occurrences for normalization
            int total = 0;
            for (int count : SUM_STRIPES.values()) {
                total += count;
            }

            // Emit total count for current word (with empty second word)
            BIGRAM.set(key.toString(), "");
            FREQ.set(total);
            context.write(BIGRAM, FREQ);

            // Calculate and emit relative frequencies
            for (Map.Entry<String, Integer> entry : SUM_STRIPES.entrySet()) {
                String b = entry.getKey();
                int count = entry.getValue();
                float freq = (float) count / total;
                BIGRAM.set(key.toString(), b);
                FREQ.set(freq);
                context.write(BIGRAM, FREQ);
            }
        }
    }

    /**
     * Combiner class to perform local aggregation of stripes.
     * Input/Output: Same as Mapper (Text, HashMapStringIntWritable)
     */
    private static class MyCombiner extends Reducer<Text, HashMapStringIntWritable, Text, HashMapStringIntWritable> {
        private final static HashMapStringIntWritable SUM_STRIPES = new HashMapStringIntWritable();

        @Override
        public void reduce(Text key, Iterable<HashMapStringIntWritable> stripes, Context context)
                throws IOException, InterruptedException {
            SUM_STRIPES.clear();
            
            // Merge partial stripes locally
            for (HashMapStringIntWritable stripe : stripes) {
                for (Map.Entry<String, Integer> entry : stripe.entrySet()) {
                    SUM_STRIPES.increment(entry.getKey(), entry.getValue());
                }
            }
            context.write(key, SUM_STRIPES);
        }
    }

    // Constructor
    public BigramFrequencyStripes() {
    }

    // Configuration constants
    private static final String INPUT = "input";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    /**
     * Main job configuration and execution method.
     * @param args Command line arguments
     * @return Job execution status (0 for success)
     */
    @Override
    public int run(String[] args) throws Exception {
        // Configure command line options
        Options options = new Options();
        options.addOption(OptionBuilder.withArgName("path").hasArg().withDescription("input path").create(INPUT));
        options.addOption(OptionBuilder.withArgName("path").hasArg().withDescription("output path").create(OUTPUT));
        options.addOption(OptionBuilder.withArgName("num").hasArg().withDescription("number of reducers").create(NUM_REDUCERS));

        // Parse command line arguments
        CommandLine cmdline;
        CommandLineParser parser = new GnuParser();
        try {
            cmdline = parser.parse(options, args);
        } catch (ParseException exp) {
            System.err.println("Error parsing command line: " + exp.getMessage());
            return -1;
        }

        // Validate required arguments
        if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setWidth(120);
            formatter.printHelp(this.getClass().getName(), options);
            ToolRunner.printGenericCommandUsage(System.out);
            return -1;
        }

        // Get configuration values
        String inputPath = cmdline.getOptionValue(INPUT);
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        // Configure Hadoop job
        Configuration conf = getConf();
        Job job = Job.getInstance(conf);
        job.setJobName(BigramFrequencyStripes.class.getSimpleName());
        job.setJarByClass(BigramFrequencyStripes.class);
        job.setNumReduceTasks(reduceTasks);

        // Set input/output paths
        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        // Set MapReduce types
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(HashMapStringIntWritable.class);
        job.setOutputKeyClass(PairOfStrings.class);
        job.setOutputValueClass(FloatWritable.class);

        // Set job classes
        job.setMapperClass(MyMapper.class);
        job.setCombinerClass(MyCombiner.class);
        job.setReducerClass(MyReducer.class);

        // Delete output directory if exists
        FileSystem.get(conf).delete(new Path(outputPath), true);

        // Execute job and measure time
        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return 0;
    }

    /**
     * Main entry point for command line execution.
     * @param args Command line arguments
     */
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new BigramFrequencyStripes(), args);
    }
}
