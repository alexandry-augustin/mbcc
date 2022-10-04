using System;
using System.Collections.Generic;
using MicrosoftResearch.Infer.Distributions;
using utils.mbcc;
using System.Linq;
using MicrosoftResearch.Infer.Maths;

namespace models.linop
{
	public class LinOp
	{
		/// <summary>
		/// Compute linear opinion pool (LinOp) of each task from a list of multi-class data 
		/// </summary>
		/// <param name="taskIndices">Task indices.</param>
		/// <param name="workerDists">Worker dists.</param>
		public static Vector[] Infer(int[][] taskIndices, Vector[][] workerDists)
		{
			Vector[] ret = new Vector[taskIndices.Length];
			return ret;
		}
		/// <summary>
		/// Compute linear opinion pool (LinOp) of each task from a list of multi-class data 
		/// </summary>
		public static Dictionary<string, Dirichlet> Infer(IList<utils.mbcc.DatumDist> data, int nbLabels)
		{
			Console.WriteLine("--- LinOp ---");
			Dictionary<string, Dirichlet> ret = new Dictionary<string, Dirichlet>();
			var taskGroup = data.GroupBy(d => d.TaskId).ToArray(); //array of unique TaskIds
			foreach(var g in taskGroup)
			{
				double[] sumDist = new double[nbLabels];
				Vector[] workerDists = g.Select(d => d.WorkerDistr).ToArray(); //select all the WorkerDist of this TaskId
				foreach (Vector d in workerDists)
					sumDist = d.Select((val, i) => sumDist[i] + val).ToArray(); //sum all workers' distribution for this TaskId

				double[] avgDist = null;
				avgDist = sumDist.Select (val => val / workerDists.Count ()).ToArray (); //average the sum of workers' distribution for this TaskId

				ret[g.Key] = Dirichlet.PointMass (Vector.FromArray (avgDist));
			}

			return ret;
		}
	}
}

