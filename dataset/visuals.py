# import matplotlib.pyplot as plt
# import numpy as np

# def show_aug(inputs, nrows=3, ncols=3, image=True):
#     plt.figure(figsize=(25., 25.))
#     plt.subplots_adjust(wspace=0., hspace=0.)
#     i_ = 0
    
#     if len(inputs) > 3:
#         inputs = inputs[:3]
        
#     for idx in range(len(inputs)):
    
#         # normalization
#         if image is True:           
#             img = inputs[idx].numpy().transpose(1,2,0)
#             mean = [0.485, 0.456, 0.406]
#             std = [0.229, 0.224, 0.225] 
#             img = (img*std+mean).astype(np.float32)
#         else:
#             img = inputs[idx].numpy().astype(np.float32)
#             img = img[0,:,:]

            
#         plt.subplot(nrows, ncols, i_+1)
#         plt.imshow(img); 
#         plt.axis('off')
 
#         i_ += 1
        
#     return plt.show()
