using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Utils;
using models.ibcc;

namespace models.mbcc
{
	public class MBCCPosteriors
	{
		/// <summary>
		/// The posterior of the true label distribution of each task
		/// </summary>
		public Dirichlet[] TrueLabelDist;
		/// <summary>
		/// The Dirichlet parameters of the confusion matrix of each worker.
		/// </summary>
		public Dirichlet[][] WorkerConfusionMatrix;
	}
}
