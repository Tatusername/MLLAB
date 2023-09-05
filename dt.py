import math 
import pandas as pd
df = pd.read_csv(r'C:\Users\Kishan\Desktop\ml_lab\traintennis.csv')

class Node:
    def __init__(self, attributes):
        """
        Creating a node with node name as attribute and creating a {decision,new_attribute/answer}
        """
        self.attributes = attributes
        self.decision = {}
        
def entropy(true_value, total):
    if (true_value)==0 or (total-true_value) == 0:# To avoid zero error 
        return 0
    else:
        entropy = -(true_value/total)*math.log((true_value/total),2)-((total-true_value)/total)*math.log((total-true_value)/total,2)
    return round(entropy,3)

def entropy_attribute(data:pd.DataFrame):
    total_gains = {}
    """
    If all samples are of same i.e All True/False, 
    value_counts() returns only one value which means we reached end of node 
    Returns:
        answer:str know we reached end else answer is a instance of Node
    """
    if len(data.PlayTennis.value_counts())==1:
        answer = data.PlayTennis.value_counts().to_dict()
        return list(answer.keys())[0]

    else:
        """
        Get gains for all attribute till we reach end of node 
        """
        true, false = data.PlayTennis.value_counts()
        overall_entropy = entropy(true,(true+false))
        for attribute in data.columns[:-1]:
            gains = 0
            labels = data[attribute].value_counts().to_dict()
            for key, total in labels.items():
                true_values = sum((data[attribute]==key) & (data['PlayTennis']=='Yes'))# Count of all true for which attribute value and PlayTennis is 'Yes'
                gains += -(total/(true+false))*entropy(true_values, total)
            total_gains[attribute] = round(gains + overall_entropy, 3)
        return total_gains

def build_tree(data:pd.DataFrame, parent:'Node'=None, decision=None):
    gains = entropy_attribute(data)# Get information gain of every attribute for a dataframe
    if isinstance(gains, str):# Return a {decision:str, answer:str} if gains is str knowing we reached of node
        parent.decision[decision] = gains
    else:
        max_gain_attribute = max(gains, key = gains.get) # find attribute with max information gain
        child = Node(max_gain_attribute)# Create a attribute to add
        if parent == None: # Create a root node at beginning 
            parent = child 
        else:
            parent.decision[decision]= child # Create a {decision:str,attribute:Node} in node parent.decision 
        for decision in data[max_gain_attribute].unique(): # Iterate through all possible decision for a attribute
            new_data = data[data[max_gain_attribute]==decision] # Forming a reduced data which contains only the rows of the deicision taken 
            if "Yes" and "No" not in parent.decision.values():# If parent.decision dict.values() doesn't have both "Yes" and "No" we further add another node
                build_tree(new_data.drop(columns = max_gain_attribute), child, decision) 

build_tree(df)
