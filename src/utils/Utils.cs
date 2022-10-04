using System;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Distributions;
using System.Collections.Generic;
using System.Linq;
using MicrosoftResearch.Infer.Utils;

namespace utils
{
	public class Utils
	{
		/// <summary>
		/// Is the point mass distribution
		/// </summary>
		/// <param name="dist">Distribution</param>
		/// <returns><c>true</c>, if distribution is a point mass, <c>false</c> otherwise.</returns>
		public static bool isPointMass(int[] dist)
		{
			if ((dist.Max ()!=0) && (dist.Sum () == dist.Max ()))
				return true;

			return false;
		}
		/// <summary>
		/// Convert type: Discrete[] -> double[,]
		/// </summary>
		/// <param name="e">E.</param>
		public static double[,] convert_ (Discrete[] e)
		{
			int n1 = e.Length;
			int n2=e [0].GetProbs().Count;
			double[,] ret=new double[n1, n2];
			for (int i = 0; i < n1; i++)
			{
				Vector temp=e [i].GetProbs();
				for (int j = 0; j < n2; j++)
					ret [i,j] = temp[j];
			}
			return ret;
		}
		/// <summary>
		/// One-of-K representation
		/// </summary>
		/// <param name="index">Index.</param>
		/// <param name="length">Length.</param>
		public static Vector OneOfK(int index, int length)
		{
//			if (index >= length)
//				throw;

			Vector ret=Vector.Zero (length);
			ret [index] = 1.0;
			return ret;
		}

		public static Vector[] Eye(int dim)
		{
			Vector[] ret=new Vector[dim];
			Vector row = null;
			for (int i = 0; i < dim; i++)
			{
				for (int j = 0; j < dim; j++)
					row=Vector.FromArray(Util.ArrayInit(dim, k => k == i ? 1.0: 0.0));
				ret[i]=row;
			}
			return ret;
		}
		/// <summary>
		/// Convert array of array in to 2D array
		/// http://stackoverflow.com/questions/26291609/converting-jagged-array-to-2d-array-c-sharp
		/// </summary>
		/// <returns>The d.</returns>
		/// <param name="source">Source.</param>
		/// <typeparam name="T">The 1st type parameter.</typeparam>
		public static T[,] To2D<T>(T[][] source)
		{
			try
			{
				int FirstDim = source.Length;
				int SecondDim = source.GroupBy(row => row.Length).Single().Key; // throws InvalidOperationException if source is not rectangular

				var result = new T[FirstDim, SecondDim];
				for (int i = 0; i < FirstDim; ++i)
					for (int j = 0; j < SecondDim; ++j)
						result[i, j] = source[i][j];

				return result;
			}
			catch (InvalidOperationException)
			{
				throw new InvalidOperationException("The given jagged array is not rectangular.");
			} 
		}
		public static double[,] To2D(Vector[] source)
		{
			try
			{
				int FirstDim = source.Length;
				int SecondDim = source.GroupBy(row => row.Count).Single().Key; // throws InvalidOperationException if source is not rectangular

				var result = new double[FirstDim, SecondDim];
				for (int i = 0; i < FirstDim; ++i)
					for (int j = 0; j < SecondDim; ++j)
						result[i, j] = source[i][j];

				return result;
			}
			catch (InvalidOperationException)
			{
				throw new InvalidOperationException("The given jagged array is not rectangular.");
			} 
		}
		/// <summary>
		/// Gets the column from a jagged array
		/// </summary>
		/// <returns>The column.</returns>
		/// <param name="column">Column.</param>
		/// <typeparam name="T">The 1st type parameter.</typeparam>
		public static T[] getColumn<T>(T[][] array, int column)
		{
			return array.Where(o => (o != null && o.Count() > column))
						.Select(o => o[column])
						.ToArray();
		}
		public static double[] getColumn<T>(Vector[] array, int column)
		{
			return array.Where(o => (o != null && o.Count() > column))
				.Select(o => o[column])
				.ToArray();
		}

//MBCC specific

		/// <summary>
		/// Repeat each element of the input array n times
		/// [1, 2, 3] => [1, 1, 2, 2, 3, 3] if n=2
		/// </summary>
		/// <param name="a">The alpha component.</param>
		/// <param name="n">N.</param>
		public static int[] Repeat(int[] a, int n)
		{
			List<int> ret=new List<int>();
			for (int j = 0; j < a.Length; j++) 
			{
				List<int> sub=Enumerable.Repeat(a[j], n).ToList();
				ret.AddRange(sub);
			}
			return ret.ToArray();
		}
		/// <summary>
		/// To the worker dist.
		/// </summary>
		/// <returns>The worker dist.</returns>
		/// <param name="workerConfusionMatrix">Worker confusion matrix.</param>
		/// <param name="trueLabelDistr">True label distr.</param>
		public static Discrete[] ToWorkerDist (Vector[][] workerConfusionMatrix, Discrete[] trueLabelDistr)
		{
			int taskCount=trueLabelDistr.Length;
			int workerCount = workerConfusionMatrix.Length;
			Discrete[] workerDist = new Discrete[taskCount];
			//TODO
			return workerDist;
		}
		/// <summary>
		/// Build worker distributions from samples
		/// </summary>
		/// <returns>The to worker dists.</returns>
		/// <param name="taskIndex">Task index.</param>
		/// <param name="workerLabels">Worker labels.</param>
		/// <param name="nbTasks">Nb tasks.</param>
		/// <param name="nbWorkers">Nb workers.</param>
		public static Vector[][] SamplesToWorkerDists(
			int[][] taskIndex, 
			int[][] workerLabels, 
			int nbTasks, 
			int nbWorkers, 
			int nbLabels)
		{
			Vector[][] workersDist = new Vector[nbWorkers][];
			for (int w=0; w < nbWorkers; w++)
			{
				workersDist[w]=new Vector[nbTasks];
				for (int t = 0; t < nbTasks; t++)
					workersDist[w][t]=Vector.Zero (nbLabels);
				for (int i = 0; i < taskIndex [w].Length; i++)
				{
					int taskIdx = taskIndex [w] [i];
					int workerLbl = workerLabels [w] [i];
					workersDist [w] [taskIdx] [workerLbl] += 1;
				}
			}
			//normalise to distribution
			for (int w=0; w < workersDist.Length; w++)
				for (int t=0; t < workersDist[w].Length; t++)
				{
					double sum = workersDist [w] [t].Sum ();
					if(sum>0)
						workersDist [w] [t] = Vector.FromArray(workersDist [w] [t].Select(e => (double) e / sum).ToArray());
//					else
//						throw 
				}
			return workersDist;
		}
	}
}

