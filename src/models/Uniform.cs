using System;
using System.Collections.Generic;
using MicrosoftResearch.Infer.Distributions;
using utils.mbcc;
using System.Linq;

namespace infer.net
{
	public class Uniform
	{
		/// <summary>
		/// Infer 
		/// </summary>
		/// <param name="data">Data.</param>
		/// <param name="nbLabels">Nb labels.</param>
		public static Dictionary<string, Dirichlet> Infer(IList<DatumDist> data, int nbLabels)
		{
			Console.WriteLine("--- Uniform ---");
			Dictionary<string, Dirichlet> ret = new Dictionary<string, Dirichlet>();
			var taskGroup = data.GroupBy(d => d.TaskId).ToArray(); //array of unique TaskIds
			foreach(var g in taskGroup)
			{
				ret[g.Key] = Dirichlet.Uniform(nbLabels);
			}

			return ret;
		}
	}
}

