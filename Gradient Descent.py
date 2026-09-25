loss = criterion(outputs, y) 

loss.backward() 

optimizer.step()
