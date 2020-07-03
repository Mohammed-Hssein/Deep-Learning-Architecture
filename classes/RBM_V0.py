# Boltzmann machines


import torch

# Creating the arcitecture of the network (Probabilistic graphical model)


class  RestrictetBoltzmanMachine():
    
    def __init__(self, nv, nh):
        # initialize the weights as matrix nh, nv Gaussian dis, bieses of visible and hidden nodes
        self.W = torch.randn(nh, nv) 
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
        self.NB_VISIBLE_NODES = nv
        self.NB_HIDDEN_NODES = nh
        self.TRAINING_SAVING = None
        self.LOSS = None

    def  sample_h(self, x): # x corresponds to visible neurons
        """
        the sample will activate the hidden nodes in a certain probability using 
        probability distribution p(h = 1|v)
        
        p_h_given_v is a vector of probabilities for the neurons hidden to be activated. 
    	To sample each one (treshholder for instance ) we use bernouilli distribution
    	It works by picking a random number between 0 and 1 : if the i'th element is below, then
    	neron will be activated.
        """
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx) # make sure bias is applied to each line of wx
        p_h_given_v = torch.sigmoid(activation) # n elements nh probability to be activated

        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def  sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation) 
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    

    def  update_weights(self, v0, vk, ph0, phk):
        """
		v0 : vector of visible nodes step 0 (v0_0)
		v0 : vector of visible nodes obtianed step K (v0_k)
		ph0: vector of probabilities of hidden nodes to be activated at step 0 p(h=1|v0_0)
		phk: vector of probabilities of hidden nodes obtained at step k p(h=1|v0_k)
        
        Here we build in k steps the contrastive divergence technique to approach the gradient
        in order to maximize the likelihood. We use gibbs sampling in order to converge the 
        distribution to the desired distribution of the training set. This is guaranteed by 
        markov chain process.
        """
        
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
    
    def contrastive_divergence(self, training_set, steps, batch_size, start_index):
        """Docstring : 
            compure Contrastive divergence to find an approximation of weights and biases in steps
            
            returns v0 , vK, ph0, phk, respectively
        """
        vk = training_set[start_index:start_index+batch_size]
        v0 = training_set[start_index:start_index+batch_size]
        ph0,_ = self.sample_h(v0)
        
        for k in range(steps):
            _,hk = self.sample_h(vk)
            _,vk = self.sample_v(hk)
            vk[v0<0] = v0[v0<0] # juste in case of .... datasets aren't always perfect !
            
        phk,_ = self.sample_h(vk)
        
        return v0, vk, ph0, phk
        
        
    
    def train(self, training_set, steps, batch_size, epochs, loss="mean"):
        """
        launch training of the rbm object
        """
        self.copy_training(training_set, loss)
        
        for epoch in range(1, epochs+1):
            train_loss=0
            s=0.
            for element in range(0, self.NB_VISIBLE_NODES-batch_size, batch_size):
                x0, xk, ph0, phk = self.contrastive_divergence(self.TRAINING_SAVING, steps, batch_size, element)
                self.update_weights(x0, xk, ph0, phk)
                train_loss += torch.mean(torch.abs(x0[x0>=0] - xk[x0>=0]))
                s+=1
                
            print('epoch:  '+str(epoch)+' ' + 'loss:  '+str(train_loss/s))
    
    def copy_training(self, training_set, loss):
        """
        copy of training set 
        
        TO OPTIMIZE ......
        """
        self.TRAINING_SAVING = training_set
        self.LOSS = loss
    
    def test(self, test_set):
        """
        HOW TO USE LOSS OF TRAINIG IN TESTING
        """
        test_loss = 0
        s = 0.
        for element in range(self.NB_VISIBLE_NODES):
            v = self.TRAINING_SAVING[element:element+1]
            vt = test_set[element:element+1]
            if len(vt[vt>=0]) > 0:
                _,h = self.sample_h(v)
                _,v = self.sample_v(h)
                test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
                s += 1.
        print('test loss: '+str(test_loss/s))
        

    