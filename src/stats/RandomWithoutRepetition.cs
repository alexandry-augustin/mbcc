using System;
using System.Collections.Generic;
using MicrosoftResearch.Infer.Maths;
using System.Linq;

namespace stats
{
	/// <summary>
	/// Ramdom without repetition.
	/// Taken from http://codereview.stackexchange.com/questions/61338/generate-random-numbers-without-repetitions
	/// </summary>
	public class RandomWithoutRepetition
	{

		// Note, max is excluded here!
		public static List<int> GenerateRandom(int count, int min, int maxPlus1)
		{
			//  initialize set S to empty
			//  for J := N-M + 1 to N do
			//    T := RandInt(1, J)
			//    if T is not in S then
			//      insert T in S
			//    else
			//      insert J in S
			//
			// adapted for C# which does not have an inclusive Next(..)
			// and to make it from configurable range not just 1.

			if (maxPlus1 <= min || count < 0 || 
				// max - min > 0 required to avoid overflow
				(count > maxPlus1 - min && maxPlus1 - min > 0))
			{
				// need to use 64-bit to support big ranges (negative min, positive max)
				throw new ArgumentOutOfRangeException("Range " + min + " to " + maxPlus1 + 
					" (" + ((Int64)maxPlus1 - (Int64)min) + " values), or count " + count + " is illegal");
			}

			// generate count random values.
			HashSet<int> candidates = new HashSet<int>();

			// start count values before max, and end at max
			for (int top = maxPlus1 - count; top < maxPlus1; top++)
			{
				// May strike a duplicate.
				// Need to add +1 to make inclusive generator
				// +1 is safe even for MaxVal max value because top < max
				if (!candidates.Add(Rand.Int(min, top + 1))) {
					// collision, add inclusive max.
					// which could not possibly have been added before.
					candidates.Add(top);
				}
			}

			// load them in to a list, to sort
			List<int> result = candidates.ToList();

			// shuffle the results because HashSet has messed
			// with the order, and the algorithm does not produce
			// random-ordered results (e.g. max-1 will never be the first value)
			for (int i = result.Count - 1; i > 0; i--)
			{  
				int k = Rand.Int(0, i + 1);  
				int tmp = result[k];  
				result[k] = result[i];  
				result[i] = tmp;  
			}  
			return result;
		}
	}
}

