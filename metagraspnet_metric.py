__author__ = 'y2863'
import numpy as np

TOP_LAYER = 0
SECONDARY_LAYER = 1
INTERLOCK_LAYER = 2
OTHERS_LAYER = 3

def metaGraspWeightedScoreBase(instance_scores, instance_occlusions, instance_layer_labels, layer_selection_list):
    
    assert isinstance(instance_scores, np.ndarray), 'Only numpy array is allowed'
    assert isinstance(instance_occlusions, np.ndarray), 'Only numpy array is allowed'
    assert isinstance(instance_layer_labels, np.ndarray), 'Only numpy array is allowed'
    assert isinstance(layer_selection_list, list), 'Only list is allowed'

    weight = 1.-instance_occlusions
    
    final_mask = np.zeros((instance_scores.shape[0]))
    for layer in layer_selection_list:
        current_mask = instance_layer_labels == layer
        final_mask = np.logical_or(final_mask, current_mask)

    #convert to float before operation
    final_mask = final_mask.astype(np.float)

    numerator = np.multiply(np.multiply(weight, final_mask), instance_scores)
    denominator = np.sum(np.multiply(weight, final_mask))


    final_score = np.sum(np.divide(numerator, denominator))

    return final_score


def metaGraspWeightedScore(instance_scores, instance_occlusions, instance_layer_labels):
    return metaGraspWeightedScoreBase(instance_scores, instance_occlusions, instance_layer_labels, [TOP_LAYER, SECONDARY_LAYER, INTERLOCK_LAYER])

def metaGraspWeightedScoreTop(instance_scores, instance_occlusions, instance_layer_labels):
    return metaGraspWeightedScoreBase(instance_scores, instance_occlusions, instance_layer_labels, [TOP_LAYER])

def metaGraspWeightedScoreSecondary(instance_scores, instance_occlusions, instance_layer_labels):
    return metaGraspWeightedScoreBase(instance_scores, instance_occlusions, instance_layer_labels, [SECONDARY_LAYER, INTERLOCK_LAYER])




if __name__ == "__main__":
    
    
    instance_scores = [0.8, 0.9, 0.1, 0.3]
    instance_occlusions = [0., 0.1, 0.3, 0.8]
    instance_layer_labels = [TOP_LAYER, SECONDARY_LAYER, INTERLOCK_LAYER, OTHERS_LAYER]

    instance_scores = np.array(instance_scores)
    instance_occlusions = np.array(instance_occlusions)
    instance_layer_labels = np.array(instance_layer_labels)

    print("instance scores:", instance_scores)
    print("instance occlusions:", instance_occlusions)
    print("instance layer_labels:", instance_layer_labels)

    print("top layer score:", metaGraspWeightedScoreTop(instance_scores, instance_occlusions, instance_layer_labels))
    print("secondary layer score:", metaGraspWeightedScoreSecondary(instance_scores, instance_occlusions, instance_layer_labels))
    print("combined score:", metaGraspWeightedScore(instance_scores, instance_occlusions, instance_layer_labels))
    
    
