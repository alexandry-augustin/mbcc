using System;
using System.Collections.Generic;
using MicrosoftResearch.Infer.Distributions;
using utils.mbcc;
using System.Linq;
using MicrosoftResearch.Infer.Maths;

namespace models.median
{
	public class Median
	{
		/// <summary>
		/// Compute the median of each task from a list of multi-class data
		/// </summary>
		public static Dictionary<string, Dirichlet> Infer(IList<DatumDist> data, int nbLabels)
		{
			Console.WriteLine("--- Median ---");
			Dictionary<string, Dirichlet> ret = new Dictionary<string, Dirichlet>();
			var taskGroup = data.GroupBy(d => d.TaskId).ToArray(); //array of unique TaskIds
			foreach(var g in taskGroup)
			{
				double[] median = new double[nbLabels];
				Vector[] workerDists = g.Select(d => d.WorkerDistr).ToArray(); //select all the WorkerDist of this TaskId

				for(int label=0; label<workerDists[0].Count; ++label)
				{
					double[] col = utils.Utils.getColumn<double> (workerDists, label);
					median[label] = stats.Utils.Median(col);
				}

				//normalise to distribution
				for(int i=0; i<workerDists.Count(); ++i)
				{
					double sum=workerDists[i].Sum();
					workerDists[i]=Vector.FromArray(workerDists[i].Select (j => j/sum).ToArray());
				}

				ret[g.Key] = Dirichlet.PointMass (Vector.FromArray (median));
			}

			return ret;
		}
	}
}

