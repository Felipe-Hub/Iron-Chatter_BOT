import re


def clean_data(data):
    """ Takes in a list of strings (sentences) and returns a list of cleaned strings (sentences). """
    
    final_data=[]
    
    for d in data:
        data = d.lower().strip().strip('.')
        data = data.replace(' \' ', ' \' ').replace("\"", " \" ").replace("‘", " ‘ ").replace("’", " ’ ")
        data = data.replace('\\', '').replace('-', '')
        data = data.replace('[', '').replace(']', '')
        data = data.replace(',', ' , ')
        data = data.replace('.', ' , ')
        data = data.replace("i'm", "i am")
        data = data.replace("'re", " are")
        data = data.replace("he's", "he is")
        data = data.replace("she's", "she is")
        data = data.replace("it's", "it is")
        data = data.replace("that's", "that is")
        data = data.replace("what's", "what is")
        data = data.replace("how's", "how is")
        data = data.replace("here's", "here is")
        data = data.replace("there's", "there is")
        data = data.replace("'ve", " have")
        data = data.replace("'d", " would")
        data = data.replace("'ll", " will")
        data = data.replace("can't", "cannot")
        data = data.replace("won't", "will not")
        data = data.replace("n't", " not")
        data = data.replace("'bout", "about")
        data = data.replace("'til", "until")
        data = data.replace("'cause", "because")
        data = data.replace("gonna", "going to")
        data = data.replace("kinda", "kind of")
        data = data.replace("n'", "ng")
        data = re.sub("[-()#/@;:<>{}`+=~|.!?]", '', data)
        data = '<bos> ' + data + ' <eos>'
        data = data.replace("  ", " ")
        
        final_data.append(data)
        
    return final_data