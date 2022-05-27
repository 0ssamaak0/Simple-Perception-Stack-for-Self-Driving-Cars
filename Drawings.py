def draw_rectangle(image,left_eqn,right_eqn):
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    #ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    XX, YY = np.meshgrid(np.arange(0, image.shape[1]), np.arange(0, image.shape[0]))
    region_thresholds = (XX < (right_eqn[0]*YY**2 + right_eqn[1]*YY + right_eqn[2])) & \
                        (XX > (left_eqn[0]*YY**2 + left_eqn[1]*YY + left_eqn[2])) #& \
                        #(YY < (right_eqn[0]*YY**2 + right_eqn[1]*YY + right_eqn[2])) & \
                        #(YY > (left_eqn[0]*YY**2 + left_eqn[1]*YY + left_eqn[2])) 

    line_image[region_thresholds] = (0xb9,0xff,0x99) #dcffcc
    return line_image
  
  def measure_curvature_pixels(left_fit,right_fit,ploty):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    #ploty, left_fit, right_fit = generate_data()
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    meters_per_pixel = 3.7/720
    #my = 30/720
    #mx = 3.7/700
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*(y_eval)* + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])*meters_per_pixel
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0]) * meters_per_pixel
    
    return left_curverad, right_curverad
