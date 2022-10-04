using System;
using System.Collections.Generic;
using System.Linq;

namespace stats
{
	public class Utils
	{
		//Taken from http://stackoverflow.com/questions/5383937/array-data-normalization
		public static double[] Normalise(IEnumerable<double> data, double min, double max)
		{
			double dataMax = data.Max();
			double dataMin = data.Min();
			double range = dataMax - dataMin;

			return data
				.Select(d => (d - dataMin) / range)
				.Select(n => (double)((1 - n) * min + n * max))
				.ToArray<double>();
		}
			
		public static double Median(double[] values)
		{
			var ys = values.OrderBy(x => x).ToList();
			double mid = (ys.Count - 1) / 2.0;
			double median = (ys[(int)(mid)] + ys[(int)(mid + 0.5)]) / 2;
			return median;
		}
	}
}

