# Cart-pole-AI

It is the solving of the open AI Gym cart-pole      
The problem has two different solutions.        
1. Try to solve the problem with simply q-learning algorithm.       
2. Solving with deep q-learning algorthim whose algorithm       
explained below.
    
# Algorithm.    
    
for e in range(episodes)    
  1.reset environment   
      
  while not done:   
    2.get epsilon according max( epsilon_min, min(epsilon, 1 - log10(episode + 1) * epsilon_decay ) )   
    3.(epsilon %) get a random action else:      
      action = model.predict(state)   
    4. do the action and get ( next_state, reward, done )   
    5. save(state, action, reward, next_state, done)    
    6. state = new_state    
      
  7.0 if not optimize,keep the train as below.       
  7.1 reset x_batch and y_batch   
  7.2 set minibatch as random( memory, min( len(memory), batch_size ) )   
  7.3 for state action reward next_stete done in minibatch:   
    7.4 y_target(action) = reward if is done    
        else:   y_target(action) = reward + gamma * model.predict(next_state)   
    7.5 x_batch append state and y_batch append y_target    
  7.6 model.fit( x, y, batch_size = len(x_batch), verbose = 0 )   
  
