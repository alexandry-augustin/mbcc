using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Globalization;
using utils.mbcc;
using utils;
using models.mbcc;
using models.ibcc;
using models.linop;
using models.median;
using System.Xml;
using stats;
using MicrosoftResearch.Infer;

/// <summary>
/// To show that MBCC recover tasks with multiple label on real dataset (against benchmarks)
/// </summary>
class Program
{
	/// <summary>
	/// directory separator
	/// </summary>
	public const char dirSep='/';//Path.DirectorySeparatorChar; // '/' or '\' depending on the plateform (windows or unix)
	/// <summary>
	/// Run all experiments
	/// </summary>
	/// <param name="args">args[0]=StartClusterRun, args[1]=EndClusterRun, args[2]=SpammerRatio, args[3]=NumSamplesMultiplier</param>
    public static void Main(string[] args)
    {
		int StartClusterRun = 1; //Initial run of the cluster
		int EndClusterRun = 1; //Final run of the cluster
		string ResultsPath = @"ResultsMBCC"; //Path to the results directory
		double SpammerRatio = 0.25;
		int nbSamples = 24;
		//--------------------------------------------------------
		// Inference algorithm settings
		//--------------------------------------------------------
		//IAlgorithm algo=new ExpectationPropagation();
		IAlgorithm algo=new VariationalMessagePassing();
		//IAlgorithm algo=new GibbsSampling();
		int nbIter = 40; //number of iterations of the inference engine

		//constraint on original dataset
		int maxTask = 100; //FIXEME shouldn't be hard coded
		int maxWorkerPerTask=10; //FIXEME shouldn't be hard coded
		int maxJudgmentsPerWorker = 16;//20;

		//synthetic data specific (excluding spammers)
		int nbWorkers = 21;
		int nbLabels = 10;
		int nbTasks = 16;
		double pseudoCount = 100;
		//--------------------------------------------------------
		// Parse cmdline args (old style) -- TODO get rid of this, and use the above parsing
		//--------------------------------------------------------
        if (args.Length > 1)
        {
            StartClusterRun = int.Parse(args[0]);
            EndClusterRun = int.Parse(args[1]);
        }
        if (args.Length > 2)
        {
            SpammerRatio = Double.Parse(args[2]);
        }
        if (args.Length > 3)
        {
			nbSamples = int.Parse(args[3]);
        }
		if (args.Length > 4)
		{
			nbIter = int.Parse(args[4]);
		}

		//filter original dataset
		if (args.Length > 5)
		{
			maxTask = int.Parse(args[5]);
		}
		if (args.Length > 6)
		{
			maxWorkerPerTask = int.Parse(args[6]);
		}

		//synthetic data specific
		/*if (args.Length > 5)
		{
			nbWorkers = int.Parse(args[5]);
		}
		if (args.Length > 6)
		{
			nbLabels = int.Parse(args[6]);
		}
		if (args.Length > 7)
		{
			nbTasks = int.Parse(args[7]);
		}*/
		//--------------------------------------------------------
		Console.WriteLine("Running experiment.");
		//--------------------------------------------------------
		// Select models to run
		//--------------------------------------------------------
		Dictionary<string, RunType> models=new Dictionary<string, RunType>()
		{
			{ "Uniform", RunType.UNIFORM }, 
			{ "Median", RunType.MEDIAN }, 
			{ "LinOp", RunType.LinOp },
			{ "BCC", RunType.BCC },
			{ "MBCC", RunType.MBCC }
		};
		//--------------------------------------------------------
		// Print parameters
		//--------------------------------------------------------
//			Console.WriteLine ("Model Names: {0}", models);
		Console.WriteLine ("Ratio Spammers: {0}", SpammerRatio);
		Console.WriteLine ("Nb Samples: {0}", nbSamples);
		Console.WriteLine ("Nb Workers: {0}", nbWorkers);
		Console.WriteLine ("Nb Labels: {0}", nbLabels);
		Console.WriteLine ("Nb Tasks: {0}", nbTasks);
		Console.WriteLine ("Output directory: {0}", ResultsPath);
		Console.WriteLine ("Inference algorithm: {0}", PrettyPrint.InferenceAlgorithmStr(algo));
		Console.WriteLine ("Number of iterations of the inference algorithm: {0}", nbIter);

		DateTime start = DateTime.Now;
		Console.WriteLine ("Starting time: {0}", DateTime.Now.ToString (new CultureInfo("en-GB")));
		//--------------------------------------------------------
		// Run experiment
		//--------------------------------------------------------
		for (int run = StartClusterRun; run <= EndClusterRun; run++)
		{
			Console.WriteLine("\nCluster run {0} / {1}", run, EndClusterRun);
			//--------------------------------------------------------
			// Reset the random seed with run index
			//--------------------------------------------------------
			Rand.Restart(run);
			//--------------------------------------------------------
			// Load data
			//--------------------------------------------------------
			List<DatumDist> data = null;
			bool runSyntheticData = false;
			if(runSyntheticData)
			{
				// synthetic data
				Console.WriteLine ("Loading synthetic data...");
				data = SyntheticData(nbWorkers, nbLabels, nbTasks, pseudoCount);
			}
			else
			{
				// real data
				string semEvalDatasetPath = string.Format (".{0}data{1}SemEval2007Data{2}", dirSep, dirSep, dirSep);
				string iaprtc12DatasetPath = string.Format (".{0}data{1}iapr-tc12{2}", dirSep, dirSep, dirSep);
				string flagsDatasetPath = string.Format (".{0}data{1}flags{2}", dirSep, dirSep, dirSep);

				bool skipUniform = false;
				bool skipPointMass = false;
				Console.WriteLine ("{0}skipping uniform distributions in real dataset", skipUniform?"":"Not ");
				Console.WriteLine ("{0}skipping point-mass distributions in real dataset", skipPointMass?"":"Not ");

				//Console.WriteLine ("Loading SemEval data...");
				//data = csv.LoadSemEvalData(semEvalDatasetPath, skipUniform, skipPointMass);
				//Console.WriteLine ("Loading IAPRT-C12 data...");
				//data = csv.LoadIAPRTC12Data (iaprtc12DatasetPath);
				Console.WriteLine ("Loading Flags data...");
				data = utils.csv.LoadFlagsData (flagsDatasetPath);

				//data = FilterDataPerWorkerId (data, "ARQ4J4TLTPBNC");
				//data = FilterData(data, maxTask, maxWorkerPerTask);
				//data = FilterDataPerNbJudgements (data, maxJudgmentsPerWorker); //filter
				//data = FilterDataPerNbJudgements2 (data, maxJudgmentsPerWorker); //truncate
			}
			//--------------------------------------------------------
			// Create run directory
			//--------------------------------------------------------
			string runPath = String.Format(ResultsPath + dirSep + "Run{0}", run); //The path to the results directory including the cluster run (e.g., Results/Run1).
			Directory.CreateDirectory(runPath);
			//--------------------------------------------------------
			// Create stats directory
			//--------------------------------------------------------
			string statsPath = String.Format("stats" + dirSep + "Run{0}", run);
			Directory.CreateDirectory(statsPath);
			//--------------------------------------------------------
			// Run experiment
			//--------------------------------------------------------
			try
			{
				RunExperimentWithSpammers(
					data, 
					nbLabels,
					models, 
					SpammerRatio, 
					nbSamples, 
					runPath, 
					statsPath, 
					algo, 
					nbIter,
					maxTask,
					maxWorkerPerTask,
					maxJudgmentsPerWorker);
				//--------------------------------------------------------
				// Write data to csv file
				//--------------------------------------------------------
				DataToCsv ( 
					string.Format ("{0}{1}data.csv", statsPath, dirSep),
					data, 
					6); //FIXME don't hardcode nbLables
			}
			catch (Exception e)
			{
				Console.WriteLine (e.ToString ());
				Console.WriteLine ("Inference failed. Moving on to the next run.");
			}
		}
		//--------------------------------------------------------
		// Print timings
		//--------------------------------------------------------
		TimeSpan totalDuration = DateTime.Now-start;
		Console.WriteLine ("Completion time: {0}", DateTime.Now.ToString (new CultureInfo("en-GB")));
		Console.WriteLine ("Total duration: {0} seconds", totalDuration.TotalSeconds);
	}
	/// <summary>
	/// Limit number of documents and judgments per document in the dataset
	/// </summary>
	/// <returns>The data.</returns>
	/// <param name="data">Dataset</param>
	/// <param name="maxTask">Maximum number of documents</param>
	/// <param name="maxWorkerPerTask">Maximum number of workers per document.</param>
	public static List<DatumDist> FilterData(
		List<DatumDist> data,
		int maxTask, 
		int maxWorkerPerTask)
	{
		Console.WriteLine ("Maximum number of tasks (in original dataset): {0}", maxTask);
		Console.WriteLine ("Maximum number of workers per task (in original dataset): {0}", maxWorkerPerTask);

		List<DatumDist> new_data=new List<DatumDist>();

		var taskGroup = data.GroupBy(d => d.TaskId).Take(maxTask);
		foreach (var tg in taskGroup) // Loops over the tasks
		{
			var workerGroup = tg.GroupBy(g => g.WorkerId).Take(maxWorkerPerTask);
			foreach (var wg in workerGroup) // Loops over the workers who voted a particular task
			{
				foreach (var datumDist in wg)
				{
					new_data.Add (datumDist);
				}
			}
		}

		return new_data;
	}
	/// <summary>
	/// Filter by worker Id
	/// </summary>
	/// <returns>The data.</returns>
	/// <param name="data">Dataset</param>
	public static List<DatumDist> FilterDataPerWorkerId(
		List<DatumDist> data,
		string workerId)
	{
		List<DatumDist> new_data=new List<DatumDist>();

		foreach (var datumDist in data) // Loops over each judgment
		{
			if (datumDist.WorkerId==workerId)
			{
				new_data.Add (datumDist);
			}
		}

		return new_data;
	}
	/// <summary>
	/// Filter workers by the total number of judgments they have provided
	/// </summary>
	/// <returns>The data.</returns>
	/// <param name="data">Dataset</param>
	public static List<DatumDist> FilterDataPerNbJudgements(
		List<DatumDist> data,
		int maxJudgmentsPerWorker)
	{
		List<DatumDist> new_data=new List<DatumDist>();

		var workerGroup = data.GroupBy (g => g.WorkerId);
		foreach (var wg in workerGroup) // Loops over the workers
		{
			if (wg.Count () <= maxJudgmentsPerWorker)
				foreach (var datumDist in wg) 
				{
					new_data.Add (datumDist);
				}
		}

		Console.WriteLine ("{0} worker(s) removed who have strictly more than {1} judgments (in original dataset)", 
			workerGroup.Count()-new_data.GroupBy (g => g.WorkerId).Count(), 
			maxJudgmentsPerWorker);

		return new_data;
	}
	/// <summary>
	/// Truncate the total number of judgments each worker provided
	/// </summary>
	/// <returns>The data.</returns>
	/// <param name="data">Dataset</param>
	public static List<DatumDist> FilterDataPerNbJudgements2(
		List<DatumDist> data,
		int maxJudgmentsPerWorker)
	{
		List<DatumDist> new_data=new List<DatumDist>();

		Dictionary<string, int> nbTaskRemoved = new Dictionary<string, int>();

		var workerGroup = data.GroupBy (g => g.WorkerId);
		foreach (var wg in workerGroup) // Loops over the workers
		{
			var maxWg = wg.Take (maxJudgmentsPerWorker);
			nbTaskRemoved[wg.Key]=wg.Count()-maxWg.Count();
			foreach (var datumDist in maxWg) 
			{
				new_data.Add (datumDist);
			}
		}

		Console.WriteLine ("{0} judgement(s)removed as they exceed the maximum nb of judgments per worker (in original dataset)", 
			nbTaskRemoved.Values.Sum(), 
			maxJudgmentsPerWorker);

		return new_data;
	}
	/// <summary>
	/// Generate synthetic data.
	/// </summary>
	/// <returns>The data.</returns>
	/// <param name="nbWorkers">Nb workers.</param>
	/// <param name="nbLabels">Nb labels.</param>
	/// <param name="nbTasks">Nb tasks.</param>
	public static List<DatumDist> SyntheticData(
		int nbWorkers, 
		int nbLabels, 
		int nbTasks,
		double pseudoCount=1.0)
	{
		//--------------------------------------------------
		// Generate true label distribution
		//--------------------------------------------------
		Vector[] trueLabelDistr = new Vector[nbTasks];
		for (int i = 0; i < nbTasks; i++)
		{
			//trueLabelDistr [i] = Dirichlet.Uniform (nbLabels).Sample ();

			//Point mass
			//int n=Rand.Int (nbLabels);
			//Vector pointMass=Vector.FromArray (Util.ArrayInit<double> (nbLabels, j => j == n ? 1.0 : 0.0));
			//trueLabelDistr [i] = Dirichlet.PointMass (pointMass).Sample ();

			//Point mass + noise
			int n=Rand.Int (nbLabels);
			Vector pointMass=Vector.FromArray (Util.ArrayInit<double> (nbLabels, j => j == n ? 1000.0 : 100.0));
			trueLabelDistr [i] = new Dirichlet(pointMass).Sample ();
		}
		//--------------------------------------------------------
		// Workers' distributions
		//--------------------------------------------------------
		Vector[][] workersDist = new Vector[nbWorkers][];
		for (int w = 0; w < nbWorkers; w++)
		{
			workersDist [w] = new Vector[nbTasks];
			for (int i = 0; i < nbTasks; i++)
			{
				Vector pseudoCounts = trueLabelDistr [i]*pseudoCount;
				Dirichlet prior = new Dirichlet (pseudoCounts);
				workersDist [w] [i] = prior.Sample ();
			}
		}
		//--------------------------------------------------------
		// Build the data
		//--------------------------------------------------------
		List<DatumDist> ret=new List<DatumDist>();
		for (int w = 0; w < nbWorkers; w++)
		{
			for (int i = 0; i < nbTasks; i++)
			{
				DatumDist datum = new DatumDist();
				datum.TaskId = string.Format("{0}", i);
				datum.WorkerId = string.Format("{0}", w);
				datum.WorkerArray=Util.ArrayInit(nbLabels, j => (int)(workersDist[w][i][j]*100));
				datum.WorkerDistr = workersDist[w][i];
				datum.GoldArr = Util.ArrayInit(nbLabels, j => (int?)(trueLabelDistr[i][j]*100));
				datum.GoldDistr = trueLabelDistr[i];
				ret.Add(datum);
			}
		}

		return ret;
	}
	/// <summary>
	/// Experiments with real data and spammers.
	/// </summary>
	public static void RunExperimentWithSpammers(
		List<DatumDist> data,
		int nbLabels,
		Dictionary<string, RunType> models,
		double SpammerRatio, 
		int nbSamples, 
		string runPath, 
		string statsPath,
		IAlgorithm algo,
		int nbIter,
		int maxTask,
		int maxWorkerPerTask,
		int maxJudgmentsPerWorker)
	{
		//--------------------------------------------------------
		// Define spamming strategy and add spammers
		// FIXME: nbLabels must match exactly with what's in the dataset
		//--------------------------------------------------------
		Console.WriteLine("Ratio of spammers: {0}", SpammerRatio);
		SpammingStrategy spammingStrategy = SpammingStrategy.SparesUniformWithPrior;
		Dictionary<string, Vector[]> SpammersConfusionMatrixGold = null;
		Console.WriteLine("Spamming strategy: {0}", PrettyPrint.SpammingStrategyStr(spammingStrategy));
		switch (spammingStrategy)
		{
		case SpammingStrategy.SingleLabel:
			SpammersConfusionMatrixGold=AddSingleRandomLabelSpammersToData(data, nbLabels, SpammerRatio, 0, 1.0);
			break;
		case SpammingStrategy.Uniform:
			SpammersConfusionMatrixGold = AddUniformSpammersToData (data, nbLabels, SpammerRatio);
			break;
		case SpammingStrategy.UniformWithPrior:
			double pseudoCount = 1;
			SpammersConfusionMatrixGold=AddUniformWithPriorSpammersToData(data, nbLabels, pseudoCount, SpammerRatio);
			break;
		case SpammingStrategy.SparesUniformWithPrior:
			double pseudoCount__ = 1;
			SpammersConfusionMatrixGold = AddSparseUniformWithPriorSpammersToData (data, nbLabels, pseudoCount__, SpammerRatio, maxJudgmentsPerWorker);
			break;
		case SpammingStrategy.Perfect:
			SpammersConfusionMatrixGold=AddPerfectSpammersToData(data, nbLabels, SpammerRatio);
			break;
		case SpammingStrategy.PerfectWithPrior:
			double pseudoCount_ = 1;
			SpammersConfusionMatrixGold=AddUniformWithPriorSpammersToData(data, nbLabels, pseudoCount_, SpammerRatio);
			break;
		default:
			//Test_MBCC.ReplaceWorkersWithSpammers (data); Dictionary<string, Vector[]> SpammersConfusionMatrixGold = null;
			break;
		}
		int nbSpammersPerTask=SpammersConfusionMatrixGold.Count;
		//--------------------------------------------------------
		// Sample data
		//--------------------------------------------------------
		DataMappingMultiClass mapping = new DataMappingMultiClass(data);
		Console.WriteLine("Number of samples: {0}", nbSamples);
		List<Datum> sampledData = mapping.SampleMultiClassData(mapping.Data, nbSamples);
		Dictionary<string, Vector> goldDist = mapping.GetGoldDistrPerTaskId();
		int[][] taskIndices = mapping.GetTaskIndicesPerWorkerIndex(sampledData);
		int[][] workerLabels = mapping.GetLabelsPerWorkerIndex(sampledData);
		//--------------------------------------------------------
		// Run Models
		//--------------------------------------------------------
		ErrorType errorType = ErrorType.EucledianDistance;
		foreach (var kvp in models)
		{
			string modelName = kvp.Key;
			RunType modelType = kvp.Value;

			Results results=null;
			Dictionary<string, double> TrueLabelDistributionError=null;
			Dictionary<string, double> WorkerConfusionMatrixError=null;
			DateTime start = DateTime.Now;
			try
			{
				switch (modelType)
				{
				case RunType.LinOp:
					results=new Results();
					results.TrueLabelDistribution = LinOp.Infer (mapping.Data, mapping.LabelCount);

					TrueLabelDistributionError=Divergence.ComputeTrueLabelError(goldDist, results.TrueLabelDistribution, errorType);
					break;
				case RunType.BCC:
					results=RunIBCC(mapping, sampledData, taskIndices, workerLabels, algo, nbIter);

					TrueLabelDistributionError=Divergence.ComputeTrueLabelError(goldDist, results.TrueLabel, errorType);
					WorkerConfusionMatrixError=Divergence.GetConfusionMatrixError(results.WorkerConfusionMatrix, SpammersConfusionMatrixGold, errorType);
					break;
				case RunType.MBCC:
					results=RunMBCC(mapping, sampledData, taskIndices, workerLabels, algo, nbIter);

					TrueLabelDistributionError=Divergence.ComputeTrueLabelError(goldDist, results.TrueLabelDistribution, errorType);
					WorkerConfusionMatrixError=Divergence.GetConfusionMatrixError(results.WorkerConfusionMatrix, SpammersConfusionMatrixGold, errorType);
					break;
				case RunType.MEDIAN:
					results=new Results();
					results.TrueLabelDistribution = Median.Infer (mapping.Data, mapping.LabelCount);

					TrueLabelDistributionError=Divergence.ComputeTrueLabelError(goldDist, results.TrueLabelDistribution, errorType);
					break;
				case RunType.UNIFORM:
					results=new Results();

					results.TrueLabel= new Dictionary<string, Discrete>();
					results.TrueLabelDistribution= new Dictionary<string, Dirichlet>();
					foreach(KeyValuePair<string, Vector> e in goldDist)
					{
						results.TrueLabel[e.Key]=Discrete.Uniform(mapping.LabelCount);
						results.TrueLabelDistribution[e.Key]=Dirichlet.Uniform(mapping.LabelCount);
					}

					results.WorkerConfusionMatrix= new Dictionary<string, Dirichlet[]>();
					foreach(KeyValuePair<string, Vector[]> e in SpammersConfusionMatrixGold)
					{
						results.WorkerConfusionMatrix[e.Key]= new Dirichlet[mapping.LabelCount];
						//results.WorkerConfusionMatrix[e.Key].Select(i => Dirichlet.Uniform(mapping.LabelCount));
						for(int i=0; i<mapping.LabelCount; i++)
							results.WorkerConfusionMatrix[e.Key][i]=Dirichlet.Uniform(mapping.LabelCount);
					}

					TrueLabelDistributionError=Divergence.ComputeTrueLabelError(goldDist, results.TrueLabelDistribution, errorType);
					WorkerConfusionMatrixError=Divergence.GetConfusionMatrixError(results.WorkerConfusionMatrix, SpammersConfusionMatrixGold, errorType);
					break;
				default:
					throw new System.ArgumentException ("Unknown Model");
				}
			}
			catch (Exception e)
			{
				Console.WriteLine (e.ToString ());
				Console.WriteLine("The inference failed on {0}.", modelName);

				continue;
			}

			TimeSpan duration = DateTime.Now-start;
			Console.WriteLine ("Inference duration for {0}: {1} seconds", modelName, duration.TotalSeconds);
			//--------------------------------------------------------
			// export results to csv
			//--------------------------------------------------------
			string filename = null;
			if(results.TrueLabel!=null)
			{
				filename=string.Format ("{0}{1}{2}_sr{3}_sm{4}_md{5}_mwpd{6}_TrueLabel.csv", 
					statsPath, 
					dirSep, 
					modelName, 
					SpammerRatio.ToString().Replace(".", ""), 
					nbSamples,
					maxTask,
					maxWorkerPerTask);
				TrueLabelToCsv (
					modelName,
					mapping,
					duration,
					results.TrueLabel, 
					filename,
					nbSamples,
					errorType,
					maxTask,
					maxWorkerPerTask,
					goldDist,
					SpammerRatio,
					TrueLabelDistributionError);
			}
			if(results.TrueLabelDistribution!=null)
			{
				filename=string.Format ("{0}{1}{2}_sr{3}_sm{4}_md{5}_mwpd{6}_TrueLabelDistribution.csv", 
					statsPath, 
					dirSep, 
					modelName, 
					SpammerRatio.ToString().Replace(".", ""), 
					nbSamples,
					maxTask,
					maxWorkerPerTask);
				TrueLabelDistributionToCsv (
					modelName,
					mapping,
					duration,
					results.TrueLabelDistribution,
					filename,
					nbSamples,
					errorType,
					maxTask,
					maxWorkerPerTask,
					goldDist,
					SpammerRatio,
					TrueLabelDistributionError);
			}
			if(results.WorkerConfusionMatrix!=null)
			{
				filename=string.Format ("{0}{1}{2}_sr{3}_sm{4}_md{5}_mwpd{6}_ConfusionMatrixPosterior.csv", 
					statsPath, 
					dirSep, 
					modelName, 
					SpammerRatio.ToString().Replace(".", ""), 
					nbSamples,
					maxTask,
					maxWorkerPerTask);
				ConfusionMatrixToCsv (
					modelName,
					mapping,
					duration,
					results.WorkerConfusionMatrix, 
					filename,
					nbSamples,
					errorType,
					maxTask,
					maxWorkerPerTask,
					SpammersConfusionMatrixGold,
					SpammerRatio,
					WorkerConfusionMatrixError);
			}
			filename = string.Format("{0}{1}{2}_sr{3}_sm{4}_md{5}_mwpd{6}.csv", 
				runPath, 
				dirSep, 
				modelName, 
				SpammerRatio.ToString().Replace(".", ""), 
				nbSamples,
				maxTask,
				maxWorkerPerTask);
			DoSnapshot(
				filename,
				modelName,
				algo, 
				nbIter,
				mapping,
				runPath, 
				SpammerRatio, 
				nbSpammersPerTask,
				spammingStrategy,
				nbSamples, 
				duration,
				errorType,
				maxTask,
				maxWorkerPerTask,
				TrueLabelDistributionError,
				WorkerConfusionMatrixError);
		}
	}
	/// <summary>
	/// Add spammers to real data
	/// Strategy: perfect distribution
	/// </summary>
	public static Dictionary<string, Vector[]> AddPerfectSpammersToData(
		IList<DatumDist> data,
		int nbLabels,
		double spammerRatio)
	{
		Dictionary<string, Vector[]> SpammersConfusionMatrixGold = new Dictionary<string, Vector[]>();

		int numJudgmentPerTask = data.GroupBy(d => d.TaskId).First().ToArray().Length; //number of judgments for each task
		int nbSpammersPerTask = (int)(spammerRatio * numJudgmentPerTask); //number of spammers per task
		Console.WriteLine("Number of spammers added per task: {0}", nbSpammersPerTask);

		for (int spammer = 0; spammer < nbSpammersPerTask; spammer++)
		{
			string workerId = "Spammer" + spammer;
			SpammersConfusionMatrixGold.Add (workerId, utils.Utils.Eye (nbLabels));

			var taskGroup = data.GroupBy(d => d.TaskId);
			foreach (var tg in taskGroup) //for each task in the dataset
			{
				DatumDist datum = new DatumDist();
				datum.TaskId = tg.Key;
				datum.WorkerId = workerId;
				datum.WorkerDistr = tg.First().GoldDistr;
				datum.GoldArr = tg.First().GoldArr;
				datum.GoldDistr = tg.First().GoldDistr;
				data.Add(datum);
			}
		}
		return SpammersConfusionMatrixGold;
	}
	/// <summary>
	/// Add spammers to real data
	/// Strategy: perfect distribution with Dirichlet prior
	/// </summary>
	public static Dictionary<string, Vector[]> AddPerfectWithPriorSpammersToData(
		IList<DatumDist> data,
		int nbLabels,
		double pseudoCount,
		double spammerRatio)
	{
		Dictionary<string, Vector[]> SpammersConfusionMatrixGold = new Dictionary<string, Vector[]>();

		int numJudgmentPerTask = data.GroupBy(d => d.TaskId).First().ToArray().Length; //number of judgments for each task
		int nbSpammersPerTask = (int)(spammerRatio * numJudgmentPerTask); //number of spammers per task
		Console.WriteLine("Number of spammers added per task: {0}", nbSpammersPerTask);

		for (int spammer = 0; spammer < nbSpammersPerTask; spammer++)
		{
			string workerId = "Spammer" + spammer;
			SpammersConfusionMatrixGold.Add (workerId, utils.Utils.Eye (nbLabels));

			var taskGroup = data.GroupBy(d => d.TaskId);
			foreach (var tg in taskGroup) //for each task in the dataset
			{
				;
				Vector pseudoCounts = Vector.FromArray(Util.ArrayInit(nbLabels, j =>  (double)tg.First ().GoldArr[j]));
				Dirichlet perfectPrior = new Dirichlet (pseudoCounts);

				DatumDist datum = new DatumDist();
				datum.TaskId = tg.Key;
				datum.WorkerId = workerId;
				datum.WorkerDistr = perfectPrior.Sample();
				datum.GoldArr = tg.First().GoldArr;
				datum.GoldDistr = tg.First().GoldDistr;
				data.Add(datum);
			}
		}
		return SpammersConfusionMatrixGold;
	}
	/// <summary>
	/// Add spammers to real data
	/// Strategy: uniform distribution
	/// </summary>
	public static Dictionary<string, Vector[]> AddUniformSpammersToData(
		IList<DatumDist> data,
		int nbLabels,
		double spammerRatio)
	{
		Dictionary<string, Vector[]> SpammersConfusionMatrixGold = new Dictionary<string, Vector[]>();

		int numJudgmentPerTask = data.GroupBy(d => d.TaskId).First().ToArray().Length; //number of judgments for each task
		int nbSpammersPerTask = (int)(spammerRatio * numJudgmentPerTask); //number of spammers per task
		Console.WriteLine("Number of spammers added per task: {0}", nbSpammersPerTask);

		//same uniform distribution for all spammers
		//in?[] spammerArray=Util.ArrayInit(nbLabels, j =>  100 / nbLabels);
		Vector spammerDistr=Vector.FromArray(Util.ArrayInit(nbLabels, j =>  1.0 / nbLabels));

		for (int spammer = 0; spammer < nbSpammersPerTask; spammer++)
		{
			string workerId = "Spammer" + spammer;
			SpammersConfusionMatrixGold.Add(workerId, Util.ArrayInit(nbLabels, v => spammerDistr)); //build spammer confusion matrix

			var taskGroup = data.GroupBy(d => d.TaskId);
			foreach (var tg in taskGroup) //for each task in the dataset
			{
				DatumDist datum = new DatumDist();
				datum.TaskId = tg.Key;
				datum.WorkerId = workerId;
				datum.WorkerDistr = spammerDistr;
				datum.GoldArr = tg.First().GoldArr;
				datum.GoldDistr = tg.First().GoldDistr;
				data.Add(datum);
			}
		}
		return SpammersConfusionMatrixGold;
	}
	/// <summary>
	/// Add spammers to real data
	/// Strategy: uniform distribution with Dirichlet prior
	/// </summary>
	public static Dictionary<string, Vector[]> AddUniformWithPriorSpammersToData(
		IList<DatumDist> data,
		int nbLabels,
		double pseudoCount,
		double spammerRatio)
	{
		Dictionary<string, Vector[]> SpammersConfusionMatrixGold = new Dictionary<string, Vector[]>();

		int numJudgmentPerTask = data.GroupBy(d => d.TaskId).First().ToArray().Length; //number of judgments for each task
		int nbSpammersPerTask = (int)(spammerRatio * numJudgmentPerTask); //number of spammers per task
		Console.WriteLine("Number of spammers added per task: {0}", nbSpammersPerTask);

		for (int spammer = 0; spammer < nbSpammersPerTask; spammer++)
		{
			Vector pseudoCounts=Vector.FromArray(Util.ArrayInit(nbLabels, j =>  pseudoCount));
			Dirichlet uniformPrior = new Dirichlet (pseudoCounts);

			string workerId = "Spammer" + spammer;
//				SpammersConfusionMatrixGold.Add(workerId, Util.ArrayInit(nbLabels, v => spammerDistr)); //build spammer confusion matrix
//FIXME this is now meaningless (since each spammer provide a different distribution for each task)
Vector spammerDistr = Vector.FromArray (Util.ArrayInit (nbLabels, j => 1.0 / nbLabels));
SpammersConfusionMatrixGold.Add (workerId, Util.ArrayInit (nbLabels, v => spammerDistr));

			var taskGroup = data.GroupBy(d => d.TaskId);
			foreach (var tg in taskGroup) //for each task in the dataset
			{
				DatumDist datum = new DatumDist();
				datum.TaskId = tg.Key;
				datum.WorkerId = workerId;
				datum.WorkerDistr = uniformPrior.Sample();
				datum.GoldArr = tg.First().GoldArr;
				datum.GoldDistr = tg.First().GoldDistr;
				data.Add(datum);
			}
		}
		return SpammersConfusionMatrixGold;
	}
	/// <summary>
	/// Add spammers to real data
	/// Strategy: uniform distribution with Dirichlet prior
	/// </summary>
	public static Dictionary<string, Vector[]> AddSparseUniformWithPriorSpammersToData(
		IList<DatumDist> data,
		int nbLabels,
		double pseudoCount,
		double spammerRatio,
		int maxJudgmentsPerSpammer)
	{
//FIXME meaningless
Dictionary<string, Vector[]> SpammersConfusionMatrixGold = new Dictionary<string, Vector[]>();

		int nbWorkers = data.GroupBy (g => g.WorkerId).Count();
		int nbSpammers = (int)(spammerRatio * nbWorkers);

		var taskGroup = data.GroupBy(d => d.TaskId);
		//int nbTasks = taskGroup.Count();
		List<string> taskIds = taskGroup.Select (k => k.Key).ToList();
		int nbTasks = taskIds.Count();

		Vector pseudoCounts = Vector.FromArray (Util.ArrayInit (nbLabels, j => pseudoCount));
		Dirichlet uniformPrior = new Dirichlet (pseudoCounts);

		for (int spammerId = 0; spammerId < nbSpammers; ++spammerId)
		{
			string spammerIdStr = "Spammer" + spammerId;

			/*foreach (int taskIdx in RamdomWithoutRepetition.GenerateRandom (maxJudgmentsPerSpammer, 0, nbTasks).ToArray ())
			{
				DatumDist datum = new DatumDist();
				datum.TaskId = taskIds[taskIdx];
				datum.WorkerId = spammerId;
				datum.WorkerDistr = uniformPrior.Sample();
				datum.GoldArr = tg.First().GoldArr;
				datum.GoldDistr = tg.First().GoldDistr;
				data.Add(datum);
			}*/
//				var test=taskGroup.Select( g => g.ElementAt(Rand.Int(0, g.Count())) ).Take(maxJudgmentsPerSpammer).ToList(); //select maxJudgmentsPerSpammer number of tasks randomly

			taskGroup=taskGroup.OrderBy (t => Rand.Int(0, t.Count())); //shuffle the tasks
			foreach (var task in taskGroup.Take (maxJudgmentsPerSpammer))
			{
				DatumDist datum = new DatumDist();
				datum.TaskId = task.Key;
				datum.WorkerId = spammerIdStr;
				datum.WorkerDistr = uniformPrior.Sample();
				datum.GoldArr = task.First().GoldArr;
				datum.GoldDistr = task.First().GoldDistr;
				data.Add(datum);
			}
		}

		return SpammersConfusionMatrixGold;
	}
	/// <summary>
	/// Add spammers to real data (original)
	// Strategy: point mass distributions
	/// </summary>
	public static Dictionary<string, Vector[]> AddSingleRandomLabelSpammersToData(
		IList<DatumDist> data,
		int nbLabels,
		double spammerRatio,
		int strategy,
		double spamValue = 1.0)
	{
		Dictionary<string, Vector[]> SpammersConfusionMatrixGold = new Dictionary<string, Vector[]>();

		int numJudgmentPerTask = data.GroupBy(d => d.TaskId).First().ToArray().Length; //number of judgments for each task
		int nbSpammersPerTask = (int)(spammerRatio * numJudgmentPerTask); //number of spammers per task
		//int nbSpammersPerTask =(int) (mapping.WorkerCount * spammerRatio);
		Console.WriteLine("Number of spammers added per task: {0}", nbSpammersPerTask);

		//pre-generate spammer distributions (on a single row)
		Vector[] spammerDist = new Vector[nbLabels];
		//int[] columnIdx = RamdomWithoutRepetition.GenerateRandom(3, 0, 6).ToArray();
		for (int i = 0; i < nbLabels; i++)
		{
			//spammerDist[i] = Vector.FromArray(Util.ArrayInit(mapping.LabelCount, j => 1.0/mapping.LabelCount));
			spammerDist[i] = Vector.FromArray(Util.ArrayInit(nbLabels, j => i == j ? spamValue : (1.0 - spamValue) / (nbLabels - 1)));
			//spammerDist[i] = Vector.FromArray(Util.ArrayInit(mapping.LabelCount, j => columnIdx.Contains(j) ? spamValue : (1.0 - spamValue) / (mapping.LabelCount - 2)));
		}

		//setting random row for spammers
		int[] indexSet = null;
		int numSpamLabels = 0;
		switch (strategy)
		{
		case 0:
			Console.WriteLine ("Spamming on label 0 (always)");
			break;
		case 1:
			Console.WriteLine("Spamming on a single label across all available labels");
			break;
		case 2:
			numSpamLabels = 2;
			indexSet = RandomWithoutRepetition.GenerateRandom (numSpamLabels, 0, nbLabels).ToArray (); //random rows that will be spammed
			Console.WriteLine("Spamming on a single label chosen from a set of {0} random labels", numSpamLabels);
			break;
		case 3:
			break;
		default:
			throw new System.ArgumentException ("Unknown spamming strategy");
		}

		//add spammers for each task
		var taskGroup = data.GroupBy(d => d.TaskId);
		for (int spammer = 0; spammer < nbSpammersPerTask; spammer++)
		{
			int row = 0;
			switch (strategy)
			{
			case 0:
				row = 0;
				break;
			case 1:
				row = Rand.Int(0, nbLabels);
				break;
			case 2:
				row = indexSet[Rand.Int(0, numSpamLabels)];
				break;
			case 3:
				//row = columnIdx[Rand.Int(0,3)];
				break;
			}
		
			string workerId = "Spammer" + spammer;
			SpammersConfusionMatrixGold.Add(workerId, Util.ArrayInit(nbLabels, v => spammerDist[row])); //build spammer confusion matrix

			foreach (var tg in taskGroup) //for each task in the dataset
			{
				DatumDist datum = new DatumDist();
				datum.TaskId = tg.Key;
				datum.WorkerId = workerId;
				datum.WorkerDistr = spammerDist[row];
				datum.GoldArr = tg.First().GoldArr;
				datum.GoldDistr = tg.First().GoldDistr;
				data.Add(datum);
			}
		}
		return SpammersConfusionMatrixGold;
	}
	/// <summary>
	/// Add spammers to real data
	// Strategy: two point masses distributions
	/// </summary>
	public static Dictionary<string, Vector[]> AddTwoRandomLabelSpammersToData(
		IList<DatumDist> data,
		int nbLabels,
		double spammerRatio,
		double spamValue = 1.0)
	{
		Dictionary<string, Vector[]> SpammersConfusionMatrixGold = new Dictionary<string, Vector[]>();

		int numJudgmentPerTask = data.GroupBy(d => d.TaskId).First().ToArray().Length; //number of judgments for each task
		int nbSpammersPerTask = (int)(spammerRatio * numJudgmentPerTask); //number of spammers per task
		Console.WriteLine("Number of spammers added per task: {0}", nbSpammersPerTask);

		//pre-generate spammer distributions (on a single row)
		Vector[] spammerDist = new Vector[nbLabels];
		//int[] columnIdx = RamdomWithoutRepetition.GenerateRandom(3, 0, 6).ToArray();
		for (int i = 0; i < nbLabels; i++)
		{
			//spammerDist[i] = Vector.FromArray(Util.ArrayInit(mapping.LabelCount, j => 1.0/mapping.LabelCount));
			spammerDist[i] = Vector.FromArray(Util.ArrayInit(nbLabels, j => i == j ? spamValue : (1.0 - spamValue) / (nbLabels - 1)));
			//spammerDist[i] = Vector.FromArray(Util.ArrayInit(mapping.LabelCount, j => columnIdx.Contains(j) ? spamValue : (1.0 - spamValue) / (mapping.LabelCount - 2)));
		}

		//add spammers for each task
		int numSpamLabels = 2;
		int[] indexSet=RandomWithoutRepetition.GenerateRandom(numSpamLabels, 0, nbLabels).ToArray(); //random rows that will be spammed
		var taskGroup = data.GroupBy(d => d.TaskId);
		for (int spammer = 0; spammer < nbSpammersPerTask; spammer++)
		{
			//int row = columnIdx[Rand.Int(0,3)];
			//int row = Rand.Int(0,mapping.LabelCount);
			//int row = indexSet[Rand.Int(0, numSpamLabels)];
			int row = 0;
			string workerId = "Spammer" + spammer;
			SpammersConfusionMatrixGold.Add(workerId, Util.ArrayInit(nbLabels, v => spammerDist[row])); //build spammer confusion matrix

			foreach (var tg in taskGroup) //for each task in the dataset
			{
				DatumDist datum = new DatumDist();
				datum.TaskId = tg.Key;
				datum.WorkerId = workerId;
				datum.WorkerDistr = spammerDist[row];
				datum.GoldArr = tg.First().GoldArr;
				datum.GoldDistr = tg.First().GoldDistr;
				data.Add(datum);
			}
		}
		return SpammersConfusionMatrixGold;
	}
	/// <summary>
	/// Run IBCC.
	/// </summary>
	/// <returns>The IBC.</returns>
	/// <param name="modelName">Model name.</param>
	/// <param name="mapping">Mapping.</param>
	/// <param name="GoldDistr">Gold distr.</param>
	/// <param name="model">Model.</param>
	/// <param name="numSamplesMultiplier">Number samples multiplier.</param>
	public static Results RunIBCC(
		DataMappingMultiClass mapping,
		List<Datum> sampledData,
		int[][] taskIndices,
		int[][] workerLabels,
		IAlgorithm algo,
		int nbIter)
	{
		Results ret=new Results();
		ret.WorkerConfusionMatrix=new Dictionary<string, Dirichlet[]>();
		ret.TrueLabel = new Dictionary<string, Discrete>();
		ret.TrueLabelDistribution = null;

		Console.WriteLine("--- BCC ---");
		IBCC model = new IBCC ();
		model.CreateModel(algo, nbIter);
		IBCCPosteriors posteriors = model.Infer(mapping.LabelCount, mapping.TaskCount, taskIndices, workerLabels);

		for (int w = 0; w < posteriors.WorkerConfusionMatrix.Length; w++)
		{
			ret.WorkerConfusionMatrix[mapping.WorkerIndexToId[w]] = posteriors.WorkerConfusionMatrix[w];
		}
		for (int t = 0; t < posteriors.TrueLabel.Length; t++)
		{
			ret.TrueLabel[mapping.TaskIndexToId[t]] = posteriors.TrueLabel[t];
		}

		return ret;
	}
	/// <summary>
	/// Run MBCC
	/// </summary>
	/// <returns>The MBC.</returns>
	/// <param name="modelName">Model name.</param>
	/// <param name="mapping">Mapping.</param>
	/// <param name="GoldDistr">Gold distr.</param>
	/// <param name="model">Model.</param>
	/// <param name="numSamplesMultiplier">Number samples multiplier.</param>
	public static Results RunMBCC(
		DataMappingMultiClass mapping,
		List<Datum> sampledData,
		int[][] taskIndices,
		int[][] workerLabels,
		IAlgorithm algo,
		int nbIter)
	{
		Results ret=new Results();
		ret.WorkerConfusionMatrix=new Dictionary<string, Dirichlet[]>();
		ret.TrueLabel = null;
		ret.TrueLabelDistribution = new Dictionary<string, Dirichlet>();

		Console.WriteLine("--- MBCC ---");
		MBCC model = new MBCC ();
		model.CreateModel(algo, nbIter);
		MBCCPosteriors posteriors = model.Infer(mapping.LabelCount, mapping.TaskCount, taskIndices, workerLabels);

		for (int w = 0; w < posteriors.WorkerConfusionMatrix.Length; w++)
		{
			ret.WorkerConfusionMatrix[mapping.WorkerIndexToId[w]] = posteriors.WorkerConfusionMatrix[w];
		}
		for (int t = 0; t < posteriors.TrueLabelDist.Length; t++)
		{
			ret.TrueLabelDistribution[mapping.TaskIndexToId[t]] = posteriors.TrueLabelDist[t];
		}

		return ret;
	}
	public static void DataToCsv(
		string filename, 
		List<DatumDist> data,
		int LabelCount)
	{
		Console.WriteLine ("Writting {0}", filename);

		using (StreamWriter writer = new StreamWriter (filename))
		{
			string header = "worker_id,task_id,label_count,label,worker_array,worker_distr,gold_arr,gold_distr";
			writer.WriteLine (header);

			foreach (var datum in data)
			{
				for (int label = 0; label < LabelCount; ++label)
				{
					writer.WriteLine ("{0},{1},{2},{3},{4},{5},{6},{7}", 
						datum.WorkerId,
						datum.TaskId,
						LabelCount,
						label,
						datum.WorkerArray==null?"N/A":string.Format("{0}", datum.WorkerArray[label]),
						datum.WorkerDistr==null?"N/A":string.Format("{0}", datum.WorkerDistr[label]),
						datum.GoldArr==null?"N/A":string.Format("{0}", datum.GoldArr[label]),
						datum.GoldDistr==null?"N/A":string.Format("{0}", datum.GoldDistr[label])
					);
				}
			}
		}
	}
	/// <summary>
	/// Confusions the matrix to csv.
	/// </summary>
	/// <param name="cmGold">Cm gold.</param>
	/// <param name="cm">Cm.</param>
	/// <param name="filename">Filename.</param>
	public static void ConfusionMatrixToCsv(
		string modelName, 
		DataMappingMultiClass mapping,
		TimeSpan duration,
		Dictionary<string, Dirichlet[]> cm, 
		string filename,
		int nbSamples,
		ErrorType errorType,
		int maxTask,
		int maxWorkerPerTask,
		Dictionary<string, Vector[]> cmGold=null,
		double? SpammerRatio=null,
		Dictionary<string, double> WorkerConfusionMatrixError=null)
	{
		using (StreamWriter writer = new StreamWriter (filename))
		{
			string header = "Model,RunTime(s),LabelCount,TaskCount,WorkerCount,workerId,actualLabel,predictedLabel,errorRate,errorRateGold,SpammerRatio,NumSamples,ConfusionMatrixError,ErrorType,MaxDoc,MaxWorkerPerDoc";
			writer.WriteLine (header);

			foreach (var kvp in cm)
			{
				string workerId = kvp.Key;

				for(int i=0; i < kvp.Value.Length; i++) //for each row
				{
					Vector row = kvp.Value [i].GetMean();
					Vector rowGold = (cmGold!=null && cmGold.ContainsKey(workerId)?cmGold[workerId][i]:null);
					for(int j=0; j < row.Count; j++) //for each column
					{
						writer.WriteLine ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}", 
							modelName,
							duration,
							string.Format("{0}", mapping.LabelCount),
							string.Format("{0}", mapping.TaskCount),
							string.Format("{0}", mapping.WorkerCount),
							workerId, 
							i, 
							j, 
							PrettyPrint.ToCsv (row[j]), 
							rowGold==null?"NA":PrettyPrint.ToCsv (rowGold[j]),
							SpammerRatio==null?"NA":string.Format("{0}", SpammerRatio),
							nbSamples,
							WorkerConfusionMatrixError.ContainsKey(workerId)?
								PrettyPrint.ToCsv(WorkerConfusionMatrixError[workerId])
								:"NA",
							PrettyPrint.ErrorTypeStr(errorType),
							string.Format("{0}", maxTask),
							string.Format("{0}", maxWorkerPerTask)
							);
					}
				}
			}
		}
	}
	public static void TrueLabelDistributionToCsv(
		string modelName, 
		DataMappingMultiClass mapping,
		TimeSpan duration,
		Dictionary<string, Dirichlet> TrueLabelDistribution, 
		string filename,
		int nbSamples,
		ErrorType errorType,
		int maxTask,
		int maxWorkerPerTask,
		Dictionary<string, Vector> TrueLabelDistributionGold=null,
		double? SpammerRatio=null,
		Dictionary<string, double> TrueLabelDistributionError=null)
	{
		Console.WriteLine ("Writting {0}", filename);

		using (StreamWriter writer = new StreamWriter (filename))
		{
			string header = "Model,RunTime(s),LabelCount,TaskCount,WorkerCount,taskId,label,value,valueGold,SpammerRatio,NumSamples,TrueLabelError,ErrorType,MaxDoc,MaxWorkerPerDoc";
			writer.WriteLine (header);

			foreach (var kvp in TrueLabelDistribution)
			{
				string taskId = kvp.Key;
				Vector dist = kvp.Value.GetMean ();
				Vector distGold=(TrueLabelDistributionGold!=null?TrueLabelDistributionGold[taskId]:null);
				for(int i=0; i < dist.Count; i++) //for each label
				{
					writer.WriteLine ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}", 
						modelName,
						duration,
						string.Format("{0}", mapping.LabelCount),
						string.Format("{0}", mapping.TaskCount),
						string.Format("{0}", mapping.WorkerCount),
						taskId, 
						i, 
						PrettyPrint.ToCsv (dist[i]), 
						distGold==null?"NA":PrettyPrint.ToCsv (distGold[i]),
						SpammerRatio==null?"NA":string.Format("{0}", SpammerRatio),
						nbSamples,
						PrettyPrint.ToCsv (TrueLabelDistributionError[kvp.Key]),
						PrettyPrint.ErrorTypeStr(errorType),
						string.Format("{0}", maxTask),
						string.Format("{0}", maxWorkerPerTask)
						);
				}
			}
		}
	}
	public static void TrueLabelToCsv(
		string modelName, 
		DataMappingMultiClass mapping,
		TimeSpan duration,
		Dictionary<string, Discrete> TrueLabel, 
		string filename,
		int nbSamples,
		ErrorType errorType,
		int maxTask,
		int maxWorkerPerTask,
		Dictionary<string, Vector> TrueLabelGold=null,
		double? SpammerRatio=null,
		Dictionary<string, double> TrueLabelError=null)
	{
		Console.WriteLine ("Writting {0}", filename);

		using (StreamWriter writer = new StreamWriter (filename))
		{
			string header = "Model,RunTime(s),LabelCount,TaskCount,WorkerCount,taskId,label,value,valueGold,SpammerRatio,NumSamples,TrueLabelError,ErrorType,MaxDoc,MaxWorkerPerDoc";
			writer.WriteLine (header);

			foreach (var kvp in TrueLabel)
			{
				string taskId = kvp.Key;
				Vector dist = kvp.Value.GetProbs();
				Vector distGold=(TrueLabelGold!=null?TrueLabelGold[taskId]:null);
				for(int i=0; i < dist.Count; i++) //for each label
				{
					writer.WriteLine ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}", 
						modelName,
						duration,
						string.Format("{0}", mapping.LabelCount),
						string.Format("{0}", mapping.TaskCount),
						string.Format("{0}", mapping.WorkerCount),
						taskId, 
						i, 
						PrettyPrint.ToCsv (dist[i]), 
						distGold==null?"NA":PrettyPrint.ToCsv (distGold[i]),
						SpammerRatio==null?"NA":string.Format("{0}", SpammerRatio),
						nbSamples,
						PrettyPrint.ToCsv (TrueLabelError[kvp.Key]),
						PrettyPrint.ErrorTypeStr(errorType),
						string.Format("{0}", maxTask),
						string.Format("{0}", maxWorkerPerTask)
						);
				}
			}
		}
	}
	/// <summary>
	/// Do csv snapshot
	/// </summary>
	/// <param name="modelName">Model name.</param>
	/// <param name="resultsDir">Results dir.</param>
	/// <param name="results">Results.</param>
	/// <param name="SpammerRatio">Spammer ratio.</param>
	/// <param name="numSamplesMultiplier">Number samples multiplier.</param>
	/// <param name="duration">Duration.</param>
	/// <param name="groundTruth">Ground truth.</param>
	public static void DoSnapshot(
		string filename,
		string modelName, 
		IAlgorithm algo,
		int nbIter,
		DataMappingMultiClass mapping,
		string runDir, 
		double SpammerRatio, 
		int nbSpammersPerTask,
		SpammingStrategy spammingStrategy,
		int nbSamples, 
		TimeSpan duration,
		ErrorType errorType,
		int maxTask,
		int maxWorkerPerTask,
		Dictionary<string, double> TrueLabelDistributionError=null,
		Dictionary<string, double> WorkerConfusionMatrixError=null)
	{
		Console.WriteLine ("Writting {0}", filename);

		Dictionary<string, string> csv = new Dictionary<string, string> ();

		csv["WorkerCount"]=string.Format("{0}", mapping.WorkerCount);
		csv["TaskCount"]=string.Format("{0}", mapping.TaskCount);
		csv["LabelCount"]=string.Format("{0}", mapping.LabelCount);
		csv["NumSamples"]=string.Format("{0}", nbSamples);
		csv["Model"]=modelName;
		csv["InferenceAlgorithm"]=PrettyPrint.InferenceAlgorithmStr(algo); //FIXME don't write this for LinOp
		csv["InferenceAlgoNbIter"]=string.Format("{0}", nbIter); //FIXME don't write this for LinOp
		csv["SpammerRatio"]=string.Format("{0}", SpammerRatio);
		csv["NbSpammersPerTask"]=string.Format("{0}", nbSpammersPerTask);
		csv["SpammingStrategy"]=PrettyPrint.SpammingStrategyStr(spammingStrategy);
		csv["RunTime(s)"]=string.Format("{0}", duration.TotalSeconds);
		csv["ErrorType"]=PrettyPrint.ErrorTypeStr(errorType);
		csv["MaxDoc"]=string.Format("{0}", maxTask);
		csv["MaxWorkerPerDoc"]=string.Format("{0}", maxWorkerPerTask);

		if (TrueLabelDistributionError != null)
		{
			csv ["TrueLabelTotalError"] = PrettyPrint.ToCsv (TrueLabelDistributionError.Values.Sum ());
			csv ["TrueLabelAverageError"] = PrettyPrint.ToCsv (TrueLabelDistributionError.Values.Average ());
		}
		else
		{
			csv ["TrueLabelTotalError"] = "NA";
			csv ["TrueLabelAverageError"] = "NA";
		}

		if (WorkerConfusionMatrixError != null)
		{
			//Double workerTotalError = (Double)WorkerConfusionMatrixError.Sum (i => i.Value);
			Double workerTotalError = WorkerConfusionMatrixError.Values.Sum ();
			Double averageError = WorkerConfusionMatrixError.Count == 0 ? 0 : (Double)workerTotalError / WorkerConfusionMatrixError.Count;

			csv ["ConfusionMatrixTotalError"] = PrettyPrint.ToCsv (workerTotalError);
			csv ["ConfusionMatrixAverageError"] = PrettyPrint.ToCsv (averageError);
		}
		else
		{
			csv ["ConfusionMatrixTotalError"] = "NA";
			csv ["ConfusionMatrixAverageError"] = "NA";
		}
		// write csv
		using (StreamWriter writer = new StreamWriter (filename))
		{
			writer.WriteLine (string.Join (",", csv.Keys));
			writer.WriteLine (string.Join (",", csv.Values));
		}
	}
}


