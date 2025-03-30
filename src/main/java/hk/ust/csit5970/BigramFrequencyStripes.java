package hk.ust.csit5970;

import java.io.IOException;
import java.util.*;

import org.apache.commons.cli.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

/**
 * Compute bigram frequencies using the stripes approach.
 * For each word A, creates a stripe containing frequencies of words B that follow A.
 */
public class BigramFrequencyStripes extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(BigramFrequencyStripes.class);

    /**
     * Custom Writable class extending HashMap for storing word counts
     * Key: String (neighbor word)
     * Value: IntWritable (count)
     */
    public static class HashMapStringIntWritable extends HashMap<String, IntWritable> {
        /**
         * Increments the count for a specific key
         * @param key The word to increment
         * @param value The value to add
         */
        public void increment(String key, int value) {
            if (containsKey(key)) {
                get(key).set(get(key).get() + value);
            } else {
                put(key, new IntWritable(value));
            }
        }

        /**
         * Increments the count by 1 for a specific key
         * @param key The word to increment
         */
        public void increment(String key) {
            increment(key, 1);
        }

        /**
         * Calculates the total count of all values in the map
         * @return Sum of all integer values
         */
        public int getTotalCount() {
            int total = 0;
            for (IntWritable value : values()) {
                total += value.get();
            }
            return total;
        }
    }

    /**
     * Mapper: Creates stripes for each word pair in the input
     * Input: (line_offset, line_text)
     * Output: (wordA, stripe) where stripe is a map of {wordB: count}
     */
    private static class MyMapper extends Mapper<LongWritable, Text, Text, HashMapStringIntWritable> {
        private final Text currentWord = new Text();
        private final HashMapStringIntWritable stripe = new HashMapStringIntWritable();

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String[] words = value.toString().trim().split("\\s+");
            for (int i = 0; i < words.length - 1; i++) {
                String wordA = words[i];
                String wordB = words[i + 1];
                
                currentWord.set(wordA);
                stripe.clear();
                stripe.increment(wordB);
                context.write(currentWord, stripe);
            }
        }
    }

    /**
     * Reducer: Aggregates stripes and calculates relative frequencies
     * Input: (wordA, [stripe1, stripe2,...])
     * Output: (PairOfStrings(wordA, ""), total_count)
     *         (PairOfStrings(wordA, wordB), relative_frequency)
     */
    private static class MyReducer extends Reducer<Text, HashMapStringIntWritable, PairOfStrings, FloatWritable> {
        private final HashMapStringIntWritable mergedStripe = new HashMapStringIntWritable();
        private final PairOfStrings outputKey = new PairOfStrings();
        private final FloatWritable outputValue = new FloatWritable();

        @Override
        public void reduce(Text key, Iterable<HashMapStringIntWritable> stripes, Context context)
                throws IOException, InterruptedException {
            mergedStripe.clear();
            
            // Merge all stripes for current word
            for (HashMapStringIntWritable stripe : stripes) {
                for (Map.Entry<String, IntWritable> entry : stripe.entrySet()) {
                    mergedStripe.increment(entry.getKey(), entry.getValue().get());
                }
            }

            // Calculate total occurrences
            int totalCount = mergedStripe.getTotalCount();

            // Emit total count for current word
            outputKey.set(key.toString(), "");
            outputValue.set(totalCount);
            context.write(outputKey, outputValue);

            // Emit relative frequencies for each neighbor
            for (Map.Entry<String, IntWritable> entry : mergedStripe.entrySet()) {
                String neighbor = entry.getKey();
                int count = entry.getValue().get();
                float frequency = (float) count / totalCount;
                
                outputKey.set(key.toString(), neighbor);
                outputValue.set(frequency);
                context.write(outputKey, outputValue);
            }
        }
    }

    /**
     * Combiner: Merges partial stripes locally
     * Input: (wordA, [stripe1, stripe2,...])
     * Output: (wordA, merged_stripe)
     */
    private static class MyCombiner extends Reducer<Text, HashMapStringIntWritable, Text, HashMapStringIntWritable> {
        private final HashMapStringIntWritable mergedStripe = new HashMapStringIntWritable();

        @Override
        public void reduce(Text key, Iterable<HashMapStringIntWritable> stripes, Context context)
                throws IOException, InterruptedException {
            mergedStripe.clear();
            
            for (HashMapStringIntWritable stripe : stripes) {
                for (Map.Entry<String, IntWritable> entry : stripe.entrySet()) {
                    mergedStripe.increment(entry.getKey(), entry.getValue().get());
                }
            }
            context.write(key, mergedStripe);
        }
    }

    // Driver configuration remains unchanged below this line
    // ------------------------------------------------------------------------
    
    public BigramFrequencyStripes() {}

    private static final String INPUT = "input";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    @Override
    public int run(String[] args) throws Exception {
        Options options = new Options();
        options.addOption(OptionBuilder.withArgName("path").hasArg().withDescription("input path").create(INPUT));
        options.addOption(OptionBuilder.withArgName("path").hasArg().withDescription("output path").create(OUTPUT));
        options.addOption(OptionBuilder.withArgName("num").hasArg().withDescription("number of reducers").create(NUM_REDUCERS));

        CommandLine cmdline;
        CommandLineParser parser = new GnuParser();
        try {
            cmdline = parser.parse(options, args);
        } catch (ParseException exp) {
            System.err.println("Error parsing command line: " + exp.getMessage());
            return -1;
        }

        if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setWidth(120);
            formatter.printHelp(this.getClass().getName(), options);
            ToolRunner.printGenericCommandUsage(System.out);
            return -1;
        }

        String inputPath = cmdline.getOptionValue(INPUT);
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        Configuration conf = getConf();
        Job job = Job.getInstance(conf);
        job.setJobName(BigramFrequencyStripes.class.getSimpleName());
        job.setJarByClass(BigramFrequencyStripes.class);
        job.setNumReduceTasks(reduceTasks);

        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(HashMapStringIntWritable.class);
        job.setOutputKeyClass(PairOfStrings.class);
        job.setOutputValueClass(FloatWritable.class);

        job.setMapperClass(MyMapper.class);
        job.setCombinerClass(MyCombiner.class);
        job.setReducerClass(MyReducer.class);

        FileSystem.get(conf).delete(new Path(outputPath), true);

        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return 0;
    }

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new BigramFrequencyStripes(), args);
    }
}
