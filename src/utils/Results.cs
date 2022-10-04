using System;
using MicrosoftResearch.Infer.Distributions;
using System.Collections.Generic;

namespace utils
{
	public class Results
	{
		/// <summary>
		/// The posterior of the true label for each task.
		/// </summary>
		public Dictionary<string, Discrete> TrueLabel
		{
			get;
			set;
		}
		/// <summary>
		/// The posterior of the confusion matrix of each worker.
		/// </summary>
		public Dictionary<string, Dirichlet[]> WorkerConfusionMatrix
		{
			get;
			set;
		}
		/// <summary>
		/// The posterior of the true label distribution for each task.
		/// </summary>
		public Dictionary<string, Dirichlet> TrueLabelDistribution
		{
			get;
			set;
		}
		public Results ()
		{
			TrueLabel = null;
			TrueLabelDistribution = null;
			WorkerConfusionMatrix = null;
		}
	}
}

