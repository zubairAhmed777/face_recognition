import csv

# Data to be written to the CSV
known_images = ['images.jpg', 'zubair.jpeg']
known_names = ['Robert Downey Jr.', 'Zubair Ahmed']

# Open the CSV file for writing (it will overwrite the file if it already exists)
with open('known_faces.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['image_path', 'name'])
    
    # Write the data rows
    for image, name in zip(known_images, known_names):
        writer.writerow([image, name])

print("CSV file 'known_faces.csv' has been created.")

