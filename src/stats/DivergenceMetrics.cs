using MicrosoftResearch.Infer.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using MicrosoftResearch.Infer.Distributions;
using utils;

namespace stats
{
    class Divergence
	{
        /// <summary>
		/// Computes the KL-Divergence
        /// </summary>
        /// <param name="dist1"></param>
        /// <param name="dist2"></param>
        /// <returns></returns>
		public static double KLDivergence(Vector dist1, Vector dist2)
		{
			double klDivergence = 0;
			for (int i = 0; i < dist1.Count(); i++)
			{
				if(dist1 [i]==0 || dist2 [i]==0)
					continue;

				klDivergence+=dist1 [i] * Math.Log(dist1 [i] / dist2 [i]);
			}

			return klDivergence;
		}
        /// <summary>
        /// Computes the Eucledian distance
        /// </summary>
        /// <param name="distr1"></param>
        /// <param name="distr2"></param>
        /// <returns></returns>
        public static double EucledianDistance(Vector distr1, Vector distr2)
        {
            var arr = distr1.Select((val, i) => Math.Pow(distr1[i] - distr2[i], 2)).ToArray();
            var res = Math.Sqrt(arr.Sum());
            return res;
        }
		public static double MeanSquaredError(Vector distr1, Vector distr2)
		{
			var arr = distr1.Select((val, i) => Math.Pow(distr1[i] - distr2[i], 2)).ToArray();
			var res = 1.0/arr.Length*arr.Sum();
			return res;
		}
		public static double JensenShannonDivergence(Vector dist1, Vector dist2)
		{
			double jsDivergence = 0;
			for (int i = 0; i < dist1.Count(); i++)
			{
				jsDivergence+=dist1 [i] * Math.Log(dist1 [i])+dist2 [i] * Math.Log(dist2 [i])-(dist1 [i]+dist2 [i]) * Math.Log(dist1 [i]+dist2 [i]);
			}
			return 0.5*jsDivergence;
		}
		/// <summary>
		/// Computes the Brier score (binary)
		/// </summary>
		public static double BrierScoreBinary(Vector[] estimatedLabelProb, Vector trueLabel)
		{
			double brierScore = 0;
			for (int taskIndex = 0; taskIndex < trueLabel.Count (); taskIndex++)
				for (int j = 0; j < estimatedLabelProb[0].Count; j++)
				{
					int mode = estimatedLabelProb [taskIndex].IndexOf (estimatedLabelProb [taskIndex].Max ());
					brierScore += Math.Pow (estimatedLabelProb[taskIndex][j] - (mode==j?1.0:0.0), 2);
				}
			return (1.0/trueLabel.Count ())*brierScore;
		}
		/// <summary>
		/// Computes the Brier score (multi-class)
		/// </summary>
		public static double BrierScore(Vector[] estimatedLabelProb, Vector trueLabel)
		{
			double brierScore = 0;
			Vector trueLabelRep;
			int nbLabels = estimatedLabelProb [0].Count;
			int nbTasks = trueLabel.Count ();
			for (int taskIndex = 0; taskIndex < nbTasks; taskIndex++)
			{
				trueLabelRep=utils.Utils.OneOfK ((int)trueLabel[taskIndex], nbLabels); //1-of-K representation
				for (int j = 0; j < nbLabels; j++)
				{
					brierScore += Math.Pow (estimatedLabelProb [taskIndex] [j] - trueLabelRep [j], 2);
				}
			}
			return (1.0/nbTasks)*brierScore;
		}
		/// <summary>
		/// Computes the Standard deviation.
		/// </summary>
		/// <returns>The deviation.</returns>
		/// <param name="values">Values.</param>
        public static double StandardDeviation(IEnumerable<double> values)
        {
            double avg = values.Average();
            return Math.Sqrt(values.Average(v => Math.Pow(v - avg, 2)));
        }
		/// <summary>
		/// Computes the Entropy
		/// </summary>
		/// <param name="d">D.</param>
		public static double Entropy(Discrete d)
		{
			return d == null ? double.MaxValue : -d.GetAverageLog (d); //GetAverageLog() returns 0 for 0.ln(0). Uses the natural logarithm
		}
		/// <summary>
		/// Gets the confusion matrix error.
		/// </summary>
		/// <returns>The confusion matrix error.</returns>
		/// <param name="workerConfusionMatrix">Worker confusion matrix.</param>
		/// <param name="trueConfusionMatrix">True confusion matrix.</param>
		/// <param name="errorType">Error type.</param>
		public static Dictionary<string, double> GetConfusionMatrixError(
			Dictionary<string, Dirichlet[]> workerConfusionMatrix, 
			Dictionary<string, Vector[]> trueConfusionMatrix,
			ErrorType errorType)
		{
			if (workerConfusionMatrix == null || trueConfusionMatrix == null)
				return null;

			Dictionary<string, double> ret = new Dictionary<string, double> ();
					
			foreach (var kvp in trueConfusionMatrix) //for each workerId
			{
				string workerId=kvp.Key;

				if (!workerConfusionMatrix.ContainsKey (workerId))
					continue;

				double sum = 0;
				for (int i = 0; i < kvp.Value.Count(); i++) //for each row of the confusion matrix
				{
					Vector workerRow = workerConfusionMatrix [workerId] [i].GetMean ();
					Vector trueRow = trueConfusionMatrix [workerId] [i];
					switch(errorType)
					{
					case ErrorType.EucledianDistance:
						sum += Divergence.EucledianDistance (workerRow, trueRow);
						break;
					case ErrorType.KLDivergence:
						sum += Divergence.KLDivergence (workerRow, trueRow);
						break;
					case ErrorType.MeanSquaredError:
						sum += Divergence.MeanSquaredError (workerRow, trueRow);
						break;
					}
				}
				ret[workerId]=sum;
			}
			return ret;
		}
		/// <summary>
		/// Computes the true label error (for Discrete distributions)
		/// </summary>
		/// <returns>The true label error.</returns>
		/// <param name="goldDistr">Gold distr.</param>
		/// <param name="trueLabel">True label.</param>
		/// <param name="errorType">Error type.</param>
		public static Dictionary<string, double> ComputeTrueLabelError(
			Dictionary<string, Vector> goldDistr, 
			Dictionary<string, Discrete> trueLabel, 
			ErrorType errorType)
		{
			if (goldDistr == null || trueLabel == null)
				return null;

			Dictionary<string, double> error=new Dictionary<string, double>();
			foreach(KeyValuePair<string, Vector> e in goldDistr)
			{
				switch(errorType)
				{
				case ErrorType.EucledianDistance:
					error[e.Key]=Divergence.EucledianDistance(e.Value, trueLabel[e.Key].GetProbs());
					break;
				case ErrorType.KLDivergence:
					error[e.Key]=Divergence.KLDivergence(e.Value, trueLabel[e.Key].GetProbs());
					break;
				case ErrorType.MeanSquaredError:
					error[e.Key]=Divergence.MeanSquaredError(e.Value, trueLabel[e.Key].GetProbs());
					break;
				}
			}
			return error;
		}
		/// <summary>
		/// Computes the true label error  (for Dirichlet distributions)
		/// </summary>
		/// <returns>The true label error.</returns>
		/// <param name="goldDistr">Gold distr.</param>
		/// <param name="trueLabelDistribution">True label distribution.</param>
		/// <param name="errorType">Error type.</param>
		public static Dictionary<string, double> ComputeTrueLabelError(
			Dictionary<string, Vector> goldDistr, 
			Dictionary<string, Dirichlet> trueLabelDistribution, 
			ErrorType errorType)
		{
			if (goldDistr == null || trueLabelDistribution == null)
				return null;

			Dictionary<string, double> error=new Dictionary<string, double>();
			foreach(KeyValuePair<string, Vector> e in goldDistr)
			{
				switch(errorType)
				{
				case ErrorType.EucledianDistance:
					error[e.Key]=Divergence.EucledianDistance(e.Value, trueLabelDistribution[e.Key].GetMean());
					break;
				case ErrorType.KLDivergence:
					error[e.Key]=Divergence.KLDivergence(e.Value, trueLabelDistribution[e.Key].GetMean());
					break;
				case ErrorType.MeanSquaredError:
					error[e.Key]=Divergence.MeanSquaredError(e.Value, trueLabelDistribution[e.Key].GetMean());
					break;
				}
			}
			return error;
		}
    }
}