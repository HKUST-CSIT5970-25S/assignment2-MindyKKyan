package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

/**
 * Compute the bigram correlation coefficients using "stripes" approach.
 * This implementation uses a two-pass MapReduce strategy:
 * 1. First pass: Count individual word frequencies
 * 2. Second pass: Calculate correlation coefficients using stripes pattern
 */
public class CORStripes extends Configured implements Tool {
	private static final Logger LOG = Logger.getLogger(CORStripes.class);

	/*
	 * First-pass Mapper: Count individual word frequencies
	 * Output: (word, 1) for each word occurrence
	 */
	private static class CORMapper1 extends Mapper<LongWritable, Text, Text, IntWritable> {
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			// Filter non-alphabetic characters using specified tokenizer
			String cleanDoc = value.toString().replaceAll("[^a-z A-Z]", " ");
			StringTokenizer tokenizer = new StringTokenizer(cleanDoc);
			while (tokenizer.hasMoreTokens()) {
				String word = tokenizer.nextToken().toLowerCase();
				context.write(new Text(word), new IntWritable(1));
			}
		}
	}

	/*
	 * First-pass Reducer: Sum up word frequencies
	 * Output: (word, totalFrequency)
	 */
	private static class CORReducer1 extends Reducer<Text, IntWritable, Text, IntWritable> {
		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			context.write(key, new IntWritable(sum));
		}
	}

	/*
	 * Second-pass Mapper (Stripes approach):
	 * For each line, create stripes of co-occurring words
	 * Output: (wordA, stripe) where stripe contains (wordB, 1) for each co-occurrence
	 */
	public static class CORStripesMapper2 extends Mapper<LongWritable, Text, Text, MapWritable> {
		@Override
		protected void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			// Extract unique words and sort them
			Set<String> sortedSet = new TreeSet<>();
			String cleanDoc = value.toString().replaceAll("[^a-z A-Z]", " ");
			StringTokenizer tokenizer = new StringTokenizer(cleanDoc);
			while (tokenizer.hasMoreTokens()) {
				sortedSet.add(tokenizer.nextToken().toLowerCase());
			}
			// Convert sorted words to list
			List<String> words = new ArrayList<>(sortedSet);
			// For each word A, create a stripe to count following words B (A < B to avoid duplicate counting)
			for (int i = 0; i < words.size(); i++) {
				String wordA = words.get(i);
				MapWritable stripe = new MapWritable();
				for (int j = i + 1; j < words.size(); j++) {
					Text wordB = new Text(words.get(j));
					// Increment count if word exists in stripe, otherwise set to 1
					if (stripe.containsKey(wordB)) {
						IntWritable count = (IntWritable) stripe.get(wordB);
						count.set(count.get() + 1);
					} else {
						stripe.put(wordB, new IntWritable(1));
					}
				}
				if (!stripe.isEmpty()) {
					context.write(new Text(wordA), stripe);
				}
			}
		}
	}

	/*
	 * Second-pass Combiner: Merge stripes for the same word
	 * Output: (wordA, mergedStripe) with summed co-occurrence counts
	 */
	public static class CORStripesCombiner2 extends Reducer<Text, MapWritable, Text, MapWritable> {
		@Override
		protected void reduce(Text key, Iterable<MapWritable> values, Context context)
				throws IOException, InterruptedException {
			MapWritable combined = new MapWritable();
			for (MapWritable stripe : values) {
				for (MapWritable.Entry<Writable, Writable> entry : stripe.entrySet()) {
					Text wordB = (Text) entry.getKey();
					IntWritable count = (IntWritable) entry.getValue();
					if (combined.containsKey(wordB)) {
						IntWritable existing = (IntWritable) combined.get(wordB);
						existing.set(existing.get() + count.get());
					} else {
						combined.put(new Text(wordB), new IntWritable(count.get()));
					}
				}
			}
			context.write(key, combined);
		}
	}

	/*
	 * Second-pass Reducer: Calculate correlation coefficients
	 * Uses pre-loaded word frequencies to compute COR(A,B) = Freq(A,B)/(Freq(A)*Freq(B))
	 * Output: (PairOfStrings(A,B), correlation)
	 */
	public static class CORStripesReducer2 extends Reducer<Text, MapWritable, PairOfStrings, DoubleWritable> {
		private static Map<String, Integer> word_total_map = new HashMap<String, Integer>();

		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			Path middle_result_path = new Path("mid/part-r-00000");
			Configuration middle_conf = new Configuration();
			try {
				FileSystem fs = FileSystem.get(URI.create(middle_result_path.toString()), middle_conf);
				if (!fs.exists(middle_result_path)) {
					throw new IOException(middle_result_path.toString() + " not exist!");
				}
				FSDataInputStream in = fs.open(middle_result_path);
				InputStreamReader inStream = new InputStreamReader(in);
				BufferedReader reader = new BufferedReader(inStream);
				LOG.info("Reading middle result...");
				String line = reader.readLine();
				while (line != null) {
					String[] tokens = line.split("\t");
					if(tokens.length >= 2){
						word_total_map.put(tokens[0], Integer.valueOf(tokens[1]));
					}
					line = reader.readLine();
				}
				reader.close();
				LOG.info("Finished reading middle result.");
			} catch (Exception e) {
				System.out.println(e.getMessage());
			}
		}

		@Override
		protected void reduce(Text key, Iterable<MapWritable> values, Context context)
				throws IOException, InterruptedException {
			// Merge multiple stripes for the same key
			MapWritable combined = new MapWritable();
			for (MapWritable stripe : values) {
				for (MapWritable.Entry<Writable, Writable> entry : stripe.entrySet()) {
					Text wordB = (Text) entry.getKey();
					IntWritable count = (IntWritable) entry.getValue();
					if (combined.containsKey(wordB)) {
						IntWritable existing = (IntWritable) combined.get(wordB);
						existing.set(existing.get() + count.get());
					} else {
						combined.put(new Text(wordB), new IntWritable(count.get()));
					}
				}
			}
			// Get total frequency for current key Freq(A)
			Integer freqA = word_total_map.get(key.toString());
			if (freqA == null || freqA == 0) return;
			// Calculate COR(A, B) for each B in the stripe
			for (MapWritable.Entry<Writable, Writable> entry : combined.entrySet()) {
				Text wordB = (Text) entry.getKey();
				Integer freqB = word_total_map.get(wordB.toString());
				if (freqB == null || freqB == 0) continue;
				IntWritable pairCountWritable = (IntWritable) entry.getValue();
				int pairCount = pairCountWritable.get();
				double corr = pairCount / (freqA.doubleValue() * freqB.doubleValue());
				// Output requirement A < B is satisfied as mapper ensures stripe only contains words B lexicographically after A
				PairOfStrings pair = new PairOfStrings(key.toString(), wordB.toString());
				context.write(pair, new DoubleWritable(corr));
			}
		}
	}

	/**
	 * Creates an instance of this tool.
	 */
	public CORStripes() {
	}

	private static final String INPUT = "input";
	private static final String OUTPUT = "output";
	private static final String NUM_REDUCERS = "numReducers";

	/**
	 * Runs this tool.
	 */
	@SuppressWarnings({ "static-access" })
	public int run(String[] args) throws Exception {
		Options options = new Options();

		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("input path").create(INPUT));
		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("output path").create(OUTPUT));
		options.addOption(OptionBuilder.withArgName("num").hasArg()
				.withDescription("number of reducers").create(NUM_REDUCERS));

		CommandLine cmdline;
		CommandLineParser parser = new GnuParser();

		try {
			cmdline = parser.parse(options, args);
		} catch (ParseException exp) {
			System.err.println("Error parsing command line: " + exp.getMessage());
			return -1;
		}

		// Lack of arguments
		if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
			System.out.println("args: " + Arrays.toString(args));
			HelpFormatter formatter = new HelpFormatter();
			formatter.setWidth(120);
			formatter.printHelp(this.getClass().getName(), options);
			ToolRunner.printGenericCommandUsage(System.out);
			return -1;
		}

		String inputPath = cmdline.getOptionValue(INPUT);
		String middlePath = "mid";
		String outputPath = cmdline.getOptionValue(OUTPUT);

		int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

		LOG.info("Tool: " + CORStripes.class.getSimpleName());
		LOG.info(" - input path: " + inputPath);
		LOG.info(" - middle path: " + middlePath);
		LOG.info(" - output path: " + outputPath);
		LOG.info(" - number of reducers: " + reduceTasks);

		// Setup for the first-pass MapReduce (统计单词频数)
		Configuration conf1 = new Configuration();

		Job job1 = Job.getInstance(conf1, "Firstpass");
		job1.setJarByClass(CORStripes.class);
		job1.setMapperClass(CORMapper1.class);
		job1.setReducerClass(CORReducer1.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(IntWritable.class);

		FileInputFormat.setInputPaths(job1, new Path(inputPath));
		FileOutputFormat.setOutputPath(job1, new Path(middlePath));

		// 删除已存在的中间结果目录
		Path middleDir = new Path(middlePath);
		FileSystem.get(conf1).delete(middleDir, true);

		long startTime = System.currentTimeMillis();
		job1.waitForCompletion(true);
		LOG.info("Job 1 Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

		// Setup for the second-pass MapReduce (计算 COR 值)
		Path outputDir = new Path(outputPath);
		FileSystem.get(conf1).delete(outputDir, true);

		Configuration conf2 = new Configuration();
		Job job2 = Job.getInstance(conf2, "Secondpass");
		job2.setJarByClass(CORStripes.class);
		job2.setMapperClass(CORStripesMapper2.class);
		job2.setCombinerClass(CORStripesCombiner2.class);
		job2.setReducerClass(CORStripesReducer2.class);

		job2.setOutputKeyClass(PairOfStrings.class);
		job2.setOutputValueClass(DoubleWritable.class);
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(MapWritable.class);
		job2.setNumReduceTasks(reduceTasks);

		FileInputFormat.setInputPaths(job2, new Path(inputPath));
		FileOutputFormat.setOutputPath(job2, new Path(outputPath));

		startTime = System.currentTimeMillis();
		job2.waitForCompletion(true);
		LOG.info("Job 2 Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

		return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
	 */
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new CORStripes(), args);
	}
}
