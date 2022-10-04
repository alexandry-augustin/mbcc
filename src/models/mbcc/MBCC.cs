using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Utils;
using models.ibcc;
using System.Linq;

namespace models.mbcc
{
    class MBCC: IBCC
    {
		/// <summary>
		/// The prior of the true label distribution of each task
		/// </summary>
		protected VariableArray<Dirichlet> TrueLabelDistrPrior;
        /// <summary>
        /// The true label distribution of each task
        /// </summary>
        protected VariableArray<Vector> TrueLabelDistr;
		/// <summary>
		/// Priors
		/// </summary>
		public MBCCPosteriors priors_;
		/// <summary>
		/// Defaults the priors.
		/// </summary>
		/// <returns>The priors.</returns>
		/// <param name="nbLabels">Nb labels.</param>
		/// <param name="nbTasks">Nb tasks.</param>
		/// <param name="nbWorkers">Nb workers.</param>
		protected static MBCCPosteriors DefaultPriors(int nbLabels, int nbTasks, int nbWorkers)
		{
			MBCCPosteriors priors = new MBCCPosteriors ();
			priors.TrueLabelDist = Util.ArrayInit(nbTasks, t => Dirichlet.Uniform(nbLabels));
			IBCCPosteriors bccPriors=IBCC.DefaultPriors (nbLabels, nbWorkers);
			priors.WorkerConfusionMatrix = bccPriors.WorkerConfusionMatrix;
			return priors;
		}
		/// <summary>
		/// Defines the variables and ranges.
		/// </summary>
		/// <param name="taskCount">Task count.</param>
		/// <param name="labelCount">Label count.</param>
        protected override void DefineVariablesAndRanges()
        {
			base.DefineVariablesAndRanges ();

            // The true label distribution for each task
			TrueLabelDistrPrior = Variable.Array<Dirichlet>(taskRange).Named("TrueLabelDistrPrior");
			TrueLabelDistr = Variable.Array<Vector>(taskRange).Named("TrueLabelDistr");
			TrueLabelDistr[taskRange] = Variable<Vector>.Random(TrueLabelDistrPrior[taskRange]);
			TrueLabelDistr.SetValueRange(labelRange);
        }
        /// <summary>
        /// Defines MBCC generative process.
        /// </summary>
        protected override void DefineGenerativeProcess()
        {
            // The process that generates the worker's label
			using (Variable.ForEach(workerRange))
            {
				VariableArray<Vector> trueLabelDistr = Variable.Subarray(TrueLabelDistr, taskIndices[workerRange]).Named("trueLabelDistr");
				trueLabelDistr.SetValueRange(labelRange);
                using (Variable.ForEach(kn))
                {
					Variable<int> sampledTrueLabel = Variable.Discrete(trueLabelDistr[kn]);
                    using (Variable.Switch(sampledTrueLabel))
                    {
						workerLabels[workerRange][kn] = Variable.Discrete(confusionMatrix[workerRange][sampledTrueLabel]);
                    }
                }
            }
        }
		/// <summary>
		/// Randomise the initial messages so as to break symmetry
		/// See http://research.microsoft.com/en-us/um/cambridge/projects/infernet/docs/Mixture%20of%20Gaussians%20tutorial.aspx
		/// </summary>
		protected override void RandomiseInitialMessages()
		{
			base.RandomiseInitialMessages ();
		}
		/// <summary>
		/// Infers the posteriors of BCC using the attached data.
		/// </summary>
		/// <param name="taskIndices">Task indices.</param>
		/// <param name="workerLabels">Worker labels.</param>
		public virtual MBCCPosteriors Infer(int nbLabels, int nbTasks, int[][] taskIndices, int[][] workerLabels)
        {
			this.nbTasks.ObservedValue = nbTasks;
			this.nbLabels.ObservedValue = nbLabels;
			this.nbWorkers.ObservedValue = taskIndices.Length;
			this.nbTasksPerWorker.ObservedValue = taskIndices.Select(tasks => tasks.Length).ToArray();
			this.taskIndices.ObservedValue = taskIndices;
			this.workerLabels.ObservedValue = workerLabels;

			this.priors_ = DefaultPriors(this.nbLabels.ObservedValue, nbTasks, this.nbWorkers.ObservedValue);
			this.confusionMatrixPrior.ObservedValue = priors_.WorkerConfusionMatrix;
			this.TrueLabelDistrPrior.ObservedValue = priors_.TrueLabelDist;

			RandomiseInitialMessages ();

			var posteriors = new MBCCPosteriors();
			posteriors.WorkerConfusionMatrix = engine.Infer<Dirichlet[][]>(confusionMatrix);
			posteriors.TrueLabelDist = engine.Infer<Dirichlet[]>(TrueLabelDistr);
			return posteriors;
        }
    }
}
