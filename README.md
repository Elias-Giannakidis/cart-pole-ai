# Cart-pole-AI

This is a try to solve Cart-pole AI problef     
with simply q-learning and    
because the problem has a lot of complexity     
there is a second code which solve this with    
deep q-learning algorithm   
    
# Algorithm.    
    
for e in range(episodes)    
  1.reset environment   
      
  while not done:   
    2.get epsilon according max( epsilon_min, min(epsilon, 1 - log10(episode + 1) * epsilon_decay ) )   
    3.epsilon% get a random action else:      
      action = model.predict(state)   
    4. do the action and get ( next_state, reward, done )   
    5. save(state, action, reward, next_state, done)    
    6. state = new_state    
      
  7.if not optimize, train the model like this...   
  7.1 reset x_batch and y_batch   
  7.2 set minibatch as random( memory, min( len(memory), batch_size ) )   
  7.3 for state action reward next_stete done in minibatch:   
    7.4 y_target(action) = reward if is done    
        y_target(action) = reward + gamma * model.predict(next_state)   
    7.5 x_batch append state and y_batch append y_target    
  7.6 model.fit( x, y, batch_size = len(x_batch), verbose = 0 )   
  
