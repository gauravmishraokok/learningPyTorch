import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from modules._01_utils import *

class RNN(nn.Module):
    
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size) -> None:
        super(RNN,self).__init__()
        
        self.hidden_size = hidden_size 
        self.i2h = nn.Linear(input_size + hidden_size , hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size , output_size)
        
        self.softmax = nn.LogSoftmax(dim = 1)
        
    
    def forward(self, input_tensor, hidden_tensor):
        combined_tensor = torch.cat((input_tensor, hidden_tensor),1)
        
        hidden = self.i2h(combined_tensor)
        output = self.i2o(combined_tensor)
         
        output = self.softmax(output)
        return output, hidden
    
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
category_lines, all_categories = load_data()
num_categories = len(all_categories)

rnn_model = RNN(input_size=N_LETTERS, hidden_size=256, output_size=num_categories)


def category_from_output(output):
    
    cat = output.argmax(dim=1).item()
    return all_categories[cat]


#Training part -> 

loss_fn = torch.nn.NLLLoss()

# Create the Adam optimizer
optimizer = torch.optim.RMSprop(rnn_model.parameters(), 0.0005 )



def train(line_tensor, category_tensor):
    hidden = rnn_model.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output,hidden = rnn_model(line_tensor[i], hidden)
        
    train_loss = loss_fn(output, category_tensor)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    return output, train_loss.item()

current_loss = 0
all_loss_values = []
plot_steps , print_steps = 500 , 10000 
n_iterations = 50000 


for i in tqdm(range(n_iterations)):
    category , line , category_tensor, line_tensor = random_training_example(category_lines=category_lines, all_categories=all_categories)
    
    output , loss = train(line_tensor=line_tensor, category_tensor=category_tensor)
    
    current_loss+= loss
    
    if (i+1) % plot_steps == 0:
        all_loss_values.append(current_loss / plot_steps)
        current_loss = 0
        
    if (i+1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{i+1} {(i+1)/n_iterations*100} {loss:.4f} {line} / {guess} {correct}")
    


#Plotting the loss values
plt.figure()
plt.plot(all_loss_values)
plt.show()


#Predicting on Custom Input Data
def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        
        hidden = rnn_model.init_hidden()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn_model(line_tensor[i], hidden)
        
        guess = category_from_output(output)
        print(guess)


while True:
    sentence = input("Type 'quit' to stop. Enter Input -> ")
    if sentence == "quit":
        break
    
    predict(sentence)
