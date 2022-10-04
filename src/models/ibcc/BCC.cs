using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Utils;
using System.Linq;

namespace models.ibcc
{
    /// <summary>
    /// IBCC model
    /// </summary>
    public class IBCC
    {
		/// <summary>
		/// Inference engine
		/// </summary>
		protected InferenceEngine engine;
		/// <summary>
		/// Inference algorithm.
		/// </summary>
		IAlgorithm algo;
		/// <summary>
		/// The number of iterations of the inference.
		/// </summary>
		int numberOfIterations;
		/// <summary>
		/// Priors
		/// </summary>
		public IBCCPosteriors priors;
		//--------------------------------------
		// Ranges
		//--------------------------------------
		protected Variable<int> nbTasks;
        protected Range taskRange;

		protected Variable<int> nbWorkers;
        protected Range workerRange;

		protected Variable<int> nbLabels;
        protected Range labelRange;

		protected VariableArray<int> nbTasksPerWorker;
		protected Range kn; //subset of tasks labelled by worker k (to deal with sparsity in the set of worker's labels)
		//--------------------------------------
		// Variables in the model
		//--------------------------------------		
		protected Variable<Dirichlet> labelPrortionPrior;
		protected Variable<Vector> labelProportion;
		/// <summary>
		/// The true label. t_i
		/// </summary>
        protected VariableArray<int> TrueLabel;

		protected VariableArray<VariableArray<Dirichlet>, Dirichlet[][]> confusionMatrixPrior;
        protected VariableArray<VariableArray<Vector>, Vector[][]> confusionMatrix;

		/// <summary>
		/// Array of number of tasks judged by each worker
		/// E.g. [5, 6] => worker 0 completed 5 tasks
		/// This is to get the dimension of the inner array of the jagged arrays taskIndices and workerLabels.
		/// </summary>
		protected VariableArray<VariableArray<int>, int[][]> taskIndices;
		/// <summary>
		/// The worker labels. c_i^{(k)}
		/// </summary>
		protected VariableArray<VariableArray<int>, int[][]> workerLabels;
		/// <summary>
		/// Defaults the priors.
		/// </summary>
		/// <returns>The priors.</returns>
		/// <param name="nbLabels">Nb labels.</param>
		/// <param name="nbWorkers">Nb workers.</param>
		protected static IBCCPosteriors DefaultPriors(int nbLabels, int nbWorkers)
		{
			IBCCPosteriors priors = new IBCCPosteriors ();

			priors.BackgroundLabelProb=Dirichlet.Uniform(nbLabels);

			priors.WorkerConfusionMatrix = new Dirichlet[nbWorkers][];
			for (int w = 0; w < nbWorkers; w++)
			{
				priors.WorkerConfusionMatrix[w] = new Dirichlet[nbLabels];
				for (int j = 0; j < nbLabels; j++)
				{
					priors.WorkerConfusionMatrix[w][j] = new Dirichlet(Util.ArrayInit(nbLabels, i => i == j ? 100.0: 1.0)); 
				}
			}

			return priors;
		}
        /// <summary>
        /// Initialises the ranges, the generative process and the inference engine of the BCC model.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="labelCount">The number of labels.</param>
		public virtual void CreateModel(IAlgorithm algo=null, int numberOfIterations=40)
        {
			if (algo == null)
				this.algo = new ExpectationPropagation ();
			else
				this.algo = algo;

			this.numberOfIterations = numberOfIterations;

            DefineVariablesAndRanges();
            DefineGenerativeProcess();
            DefineInferenceEngine();
        }
        /// <summary>
        /// Initializes the ranges of the variables.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="labelCount">The number of labels.</param>
		protected virtual void DefineVariablesAndRanges()
        {
			nbTasks = Variable.New<int>().Named("nbTasks");
			taskRange = new Range(nbTasks).Named("taskRange");

			nbLabels = Variable.New<int>().Named("nbLabels");
			labelRange = new Range(nbLabels).Named("labelRange");

			// label proportion
			labelPrortionPrior = Variable.New<Dirichlet>().Named("labelPrortionPrior");
			labelProportion = Variable<Vector>.Random(labelPrortionPrior).Named("labelPrortion");

			// The unobserved 'true' label for each task
			TrueLabel = Variable.Array<int>(taskRange).Attrib(QueryTypes.Marginal).Attrib(QueryTypes.MarginalDividedByPrior).Named("TrueLabel");
			TrueLabel[taskRange] = Variable.Discrete(labelProportion).ForEach(taskRange);
			labelProportion.SetValueRange(labelRange);

			nbWorkers = Variable.New<int>().Named("nbWorkers");
			workerRange = new Range(nbWorkers).Named("workerRange");

			// Confusion matrices for each worker
			confusionMatrixPrior = Variable.Array(Variable.Array<Dirichlet>(labelRange), workerRange).Named("confusionMatrixPrior");
			confusionMatrix = Variable.Array(Variable.Array<Vector>(labelRange), workerRange).Named("confusionMatrix");
			confusionMatrix[workerRange][labelRange] = Variable<Vector>.Random(confusionMatrixPrior[workerRange][labelRange]);
			confusionMatrix.SetValueRange(labelRange);

			// The tasks for each worker
			nbTasksPerWorker = Variable.Array<int>(workerRange).Named("nbTasksPerWorker");
            kn = new Range(nbTasksPerWorker[workerRange]).Named("kn");
			taskIndices = Variable.Array(Variable.Array<int>(kn), workerRange).Named("taskIndices");
            taskIndices.SetValueRange(taskRange);
            workerLabels = Variable.Array(Variable.Array<int>(kn), workerRange).Named("WorkerLabel");
        }
        /// <summary>
        /// Defines the BCC generative process.
        /// </summary>
        protected virtual void DefineGenerativeProcess()
        {
            // The process that generates the worker's label
            using (Variable.ForEach(workerRange))
            {
				VariableArray<int> trueLabel = Variable.Subarray(TrueLabel, taskIndices[workerRange]).Named("trueLabel"); //subset of TrueLabel that this worker judged
                trueLabel.SetValueRange(labelRange);
                using (Variable.ForEach(kn))
                {
                    using (Variable.Switch(trueLabel[kn]))
                    {
                        workerLabels[workerRange][kn] = Variable.Discrete(confusionMatrix[workerRange][trueLabel[kn]]);
                    }
                }
            }
        }
		/// <summary>
		/// Randomise the initial messages so as to break symmetry
		/// See http://research.microsoft.com/en-us/um/cambridge/projects/infernet/docs/Mixture%20of%20Gaussians%20tutorial.aspx
		/// </summary>
		protected virtual void RandomiseInitialMessages()
		{
			// Randomise the initial message for TrueLabel
			int nbLabels=this.nbLabels.ObservedValue;
			int nbTasks=this.nbTasks.ObservedValue;

			// Randomise the initial message for confusionMatrix
			int nbWorkers=this.nbWorkers.ObservedValue;
			Dirichlet[][] confusionMatrixInit = new Dirichlet[nbWorkers][];
			for (int w = 0; w < confusionMatrixInit.Length; w++)
			{
				confusionMatrixInit[w] = new Dirichlet[nbLabels];
				for (int j = 0; j < nbLabels; j++)
					confusionMatrixInit [w] [j] = Dirichlet.PointMass (Util.ArrayInit (nbLabels, i => i == j ? 0.75 : 0.25));
			}
			this.confusionMatrix.InitialiseTo (Distribution<Vector>.Array (confusionMatrixInit));
		}
        /// <summary>
        /// Initialises the BCC inference engine.
        /// </summary>
		protected virtual void DefineInferenceEngine()
        {
            engine = new InferenceEngine();
			engine.Algorithm = this.algo;
			engine.NumberOfIterations = this.numberOfIterations;
            engine.Compiler.UseParallelForLoops = true;
            engine.ShowProgress = true;
            engine.Compiler.WriteSourceFiles = true;
            engine.Compiler.GenerateInMemory = true;
            engine.ShowTimings = true;
            engine.ShowWarnings = false;
			engine.ShowFactorGraph = false;
        }
        /// <summary>
        /// Infers the posteriors of BCC using the attached data.
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        /// <returns></returns>
		public virtual IBCCPosteriors Infer(int nbLabels, int nbTasks, int[][] taskIndices, int[][] workerLabels)
        {
			this.nbTasks.ObservedValue = nbTasks;
			this.nbLabels.ObservedValue = nbLabels;
			this.nbWorkers.ObservedValue = taskIndices.Length;
			this.nbTasksPerWorker.ObservedValue = taskIndices.Select(tasks => tasks.Length).ToArray();
			this.taskIndices.ObservedValue = taskIndices;
			this.workerLabels.ObservedValue = workerLabels;

			this.priors = DefaultPriors(this.nbLabels.ObservedValue, this.nbWorkers.ObservedValue);
			this.labelPrortionPrior.ObservedValue = priors.BackgroundLabelProb;
			this.confusionMatrixPrior.ObservedValue = priors.WorkerConfusionMatrix;

			RandomiseInitialMessages ();

			var posteriors = new IBCCPosteriors();
            posteriors.BackgroundLabelProb = engine.Infer<Dirichlet>(labelProportion);
			posteriors.TrueLabel = engine.Infer<Discrete[]>(TrueLabel);
            posteriors.WorkerConfusionMatrix = engine.Infer<Dirichlet[][]>(confusionMatrix);

            return posteriors;
        }
    }
}