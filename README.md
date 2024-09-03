#### Project Status: First iteration complete... More to come

# Project Title: Automated Attention Tracking and Reporting.
## Project Description.
The Automated Attention Tracking and Reporting system is a simple and elegant AI-driven dashboard designed to monitor and report distraction levels during classes, meetings, or lectures in real-time. By leveraging computer vision techniques, this tool provides valuable insights into participant engagement, helping educators, trainers, and facilitators to better understand and improve the focus and attention of their audience.    

The underlying model was efficiently trained using the [Roboflow](https://roboflow.com/) platform AutoML feature, allowing seamless integration and rapid deployment. This system does not only track attention/focus but also visualizes trends over time, offering an overview of engagement patterns and distraction level growth throughout a session.   

## Users of this system get to enjoy the following benefits:
- **Real-Time Monitoring**: Instantly track and visualize distraction levels during live sessions, enabling immediate intervention to maintain focus or pre-recorded videos.
- **Comprehensive Analytics**:
  - ***Supervisor***: Access to detailed reports and trend analyses that highlight engagement patterns over time, helping to identify areas for improvement.
- **Enhanced Engagement**: By understanding distraction triggers, educators and facilitators can tailor their approach to maximize attention and participation.
- **Easy Integration**: Seamlessly incorporate the system into various educational or corporate settings with minimal setup, thanks to the model's training on the user-friendly Roboflow platform.
- **Customized distraction duration**: Set a custom threshold for the time limit before distraction is recorded e.g. 3 or 5 seconds of not paying attention = distracted.
- **Data-Driven Decisions**: Leverage the insights generated by the system to make informed decisions on how to structure sessions, optimize content delivery, and improve overall outcomes.
## Further Improvements:
- **Comprehensive Analytics**:
  - ***Supervisee***: Access to the time stamp of when the individual was distracted to enable seamless review of recorded sessions with a focus on missed portions
  - ***Supervisee***: Access to AI summary of missed portion, potentially including links to online resources to help further understand missed portions, especially in school lecture settings.
- **Customizable Alerts**: Set thresholds for distraction levels and receive real-time alerts, ensuring that attention issues are addressed promptly.
- **Distraction Levels Predictions**: Utilize data of past distraction triggers to train AI capable of predicting distraction growth levels during sessions (Will experiment with Mixture Of Experts models).
-   **Performance Prediction**: Utilize data of past distraction levels of classes (or individuals to be more granular) to predict class (or individual) performance based on attention/distraction levels throughout the term. 
  
### Tools and Libraries used:
* Plotly
* OpenCV
* Streamlit
* Roboflow AutoML
* Roboflow Data Annotation

## Snapshots of System... Demo available on [YouTube.](https://youtu.be/VviehI3x7bc?si=6o1hAmVD96Fuf14D)
![Picture1](https://github.com/user-attachments/assets/87893103-990b-4667-b89b-2c6748d08fb2)
