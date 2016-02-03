package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import org.hamcrest.core.IsInstanceOf;

import tud.ke.ml.project.framework.classifier.ANearestNeighbor;
import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but
 * probably not set)
 * 
 * @author cwirth
 *
 */
public class NearestNeighbor extends ANearestNeighbor implements Serializable {

	// private static final long serialVersionUID = 2010906213520172559L;

	protected double[] scaling;
	protected double[] translation;
	List<List<Object>> trainingsData;

	@Deprecated
	protected String[] getMatrikelNumbers() {
		return new String[] { getGroupNumber() };
	}

	@Override
	public String getGroupNumber() {
		return "43";
	}

	@Override
	protected void learnModel(List<List<Object>> data) {
		this.trainingsData = data;
	}

	protected Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> resultMap = new HashMap<Object, Double>();
		Iterator<Pair<List<Object>, Double>> subsetIterator = subset.iterator();
		while (subsetIterator.hasNext()) {
			Pair<List<Object>, Double> instanceDistancePair = subsetIterator.next();
			Object classAttributeValue = instanceDistancePair.getA().get(getClassAttribute());
			if (resultMap.containsKey(classAttributeValue)) {
				double votes = resultMap.get(classAttributeValue);
				resultMap.put(classAttributeValue, votes + 1.0);
			} else {
				resultMap.put(classAttributeValue, 1.0);
			}
		}
		return resultMap;
	}

	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> resultMap = new HashMap<Object, Double>();
		Iterator<Pair<List<Object>, Double>> subsetIterator = subset.iterator();
		while (subsetIterator.hasNext()) {
			Pair<List<Object>, Double> instanceDistancePair = subsetIterator.next();
			Object classAttributeValue = instanceDistancePair.getA().get(getClassAttribute());
			Double distanceValue = instanceDistancePair.getB();
			double weight = 1 / (distanceValue + 0.001);
			if (resultMap.containsKey(classAttributeValue)) {
				double votes = resultMap.get(classAttributeValue);
				resultMap.put(classAttributeValue, votes + weight);
			} else {
				resultMap.put(classAttributeValue, weight);
			}
		}
		return resultMap;
	}

	protected Object getWinner(Map<Object, Double> votes) {
		Object predictedClass = null;
		double highestVote = 0.0;
		Set<Object> keys = votes.keySet();
		Iterator<Object> votesIterator = keys.iterator();
		while (votesIterator.hasNext()) {
			Object key = votesIterator.next();
			double vote = votes.get(key);
			if (vote > highestVote) {
				predictedClass = key;
				highestVote = vote;
			}
		}
		return predictedClass;
	}

	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> votes;
		if (isInverseWeighting()) {
			votes = getWeightedVotes(subset);
		} else {
			votes = getUnweightedVotes(subset);
		}
		return getWinner(votes);
	}

	protected List<Pair<List<Object>, Double>> getNearest(List<Object> data) {
		List<Pair<List<Object>, Double>> distances = new LinkedList<Pair<List<Object>, Double>>();

		if (isNormalizing()) {
			double[][] normalizationArray = normalizationScaling();
			scaling = normalizationArray[1];
			translation = normalizationArray[0];

			for (int i = 0; i < data.size(); i++) {
				Object attribute = data.get(i);
				if (attribute instanceof Double) {
					double vNormalized = ((double) attribute - translation[i]) / scaling[i];
					data.set(i, vNormalized);
				}
			}
		}

		Iterator<List<Object>> it = trainingsData.iterator();
		while (it.hasNext()) {
			List<Object> neighbour = it.next();

			if (isNormalizing()) {
				for (int i = 0; i < neighbour.size(); i++) {
					Object attribute = neighbour.get(i);
					if (attribute instanceof Double) {
						double vNormalized = ((double) attribute - translation[i]) / scaling[i];
						neighbour.set(i, vNormalized);
					}
				}
			}

			Pair<List<Object>, Double> neighbourPair;
			if (getMetric() == 1) {
				neighbourPair = new Pair<List<Object>, Double>(neighbour, determineEuclideanDistance(data, neighbour));
			} else {
				neighbourPair = new Pair<List<Object>, Double>(neighbour, determineManhattanDistance(data, neighbour));
			}
			distances.add(neighbourPair);
		}

		int k = getkNearest();
		Comparator<Pair<List<Object>, Double>> cd = new Comparator<Pair<List<Object>, Double>>() {
			@Override
			public int compare(Pair<List<Object>, Double> o1, Pair<List<Object>, Double> o2) {
				return Double.compare(o1.getB(), o2.getB());
			}
		};
		distances.sort(cd);
		List<Pair<List<Object>, Double>> kNeighbours = new LinkedList<Pair<List<Object>, Double>>();
		Iterator<Pair<List<Object>, Double>> distanceIterator = distances.iterator();
		while (distanceIterator.hasNext() && k != 0) {
			kNeighbours.add(distanceIterator.next());
			k--;
		}

		return kNeighbours;
	}

	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
		Double distance = 0.0;
		int indexClassAtttribute = getClassAttribute();

		for (int i = 0; i < instance1.size(); i++) {
			if (i != indexClassAtttribute) {
				Object attributeValue1 = instance1.get(i);
				Object attributeValue2 = instance2.get(i);
				if (attributeValue1 instanceof Double) {
					distance += Math.abs((Double) attributeValue1 - (Double) attributeValue2);
				} else if (attributeValue1 instanceof String) {
					distance += attributeValue1.equals(attributeValue2) ? 0.0 : 1.0;
				} else {
					System.err.println("attribute not string or double");
				}
			}
		}

		return distance;
	}

	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
		Double distance = 0.0;
		int indexClassAtttribute = getClassAttribute();

		for (int i = 0; i < instance1.size(); i++) {
			if (i != indexClassAtttribute) {
				Object attributeValue1 = instance1.get(i);
				Object attributeValue2 = instance2.get(i);
				if (attributeValue1 instanceof Double) {
					distance += Math.pow(((Double) attributeValue1 - (Double) attributeValue2), 2);
				} else if (attributeValue1 instanceof String) {
					distance += attributeValue1.equals(attributeValue2) ? 0.0 : 1.0;
				}
			}
		}
		distance = Math.sqrt(distance);

		return distance;
	}

	protected double[][] normalizationScaling() {
		double[][] normalizationArray = new double[2][trainingsData.get(0).size()];

		if (isNormalizing()) {
			double[] minAttributes = new double[trainingsData.get(0).size()];
			double[] maxAttributes = new double[trainingsData.get(0).size()];

			for (int j = 0; j < this.trainingsData.size(); j++) {
				List<Object> attributeList = this.trainingsData.get(j);
				for (int i = 0; i < attributeList.size(); i++) {
					Object attribute = attributeList.get(i);
					if (attribute instanceof String) {
						if (j == 0) {
							minAttributes[i] = 0.0;
							maxAttributes[i] = 1.0;
						}
					} else if (attribute instanceof Double) {
						if (((double) attribute < minAttributes[i]) || (j == 0)) {
							minAttributes[i] = (double) attribute;
						} else if ((double) attribute > maxAttributes[i]) {
							maxAttributes[i] = (double) attribute;
						}
					}
				}
			}

			for (int i = 0; i < minAttributes.length; i++) {
				normalizationArray[0][i] = minAttributes[i];
				normalizationArray[1][i] = maxAttributes[i] - minAttributes[i];
				if (normalizationArray[1][i] == 0.0) {
					normalizationArray[1][i] = 1.0;
				}
			}
		}
		return normalizationArray;
	}

}
