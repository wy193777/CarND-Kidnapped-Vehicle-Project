/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"
#include <random>


std::default_random_engine ENGINE;

using namespace std;

double ParticleFilter::gaussian_random(double mean, double deviation) {
    std::normal_distribution<double> dist(mean, deviation);
    return dist(ENGINE);
}

double ParticleFilter::gaussian_probability(double mu, double sigma, double x) {
    return exp(-(pow(mu - x, 2) / pow(sigma, 2) / 2.0)) / sqrt(2.0 * M_PI * pow(sigma, 2));
}


double ParticleFilter::distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0));
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 100;
    for (int i = 0; i < num_particles; i++) {
        Particle pa = {
            i,
            gaussian_random(x, std[0]),
            gaussian_random(y, std[1]),
            gaussian_random(theta, std[2]),
            1,
        };
        this->particles.push_back(pa);
    }
    is_initialized = true;

}

void ParticleFilter::prediction(
    double delta_t, double std_pos[], double velocity, double yaw_rate) {
    for (auto particle : this->particles) {
        double pre_x = gaussian_random(particle.x, std_pos[0]);
        double pre_y = gaussian_random(particle.y, std_pos[1]);
        double pre_theta = gaussian_random(particle.theta, std_pos[2]);
        double curve = velocity / yaw_rate;
        particle.x = pre_x + curve * (sin(pre_theta + yaw_rate * delta_t) - sin(pre_theta));
        particle.y = pre_y + curve * (cos(pre_theta) - cos(pre_theta + yaw_rate * delta_t));
        particle.theta = pre_theta + yaw_rate * delta_t;
    }

}


// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. 
//       You can read more about this distribution here: 
//       https://en.wikipedia.org/wiki/Multivariate_normal_distribution
// NOTE: The observations are given in the VEHICLE'S coordinate system. 
//       Your particles are located according to the MAP'S coordinate system. 
//       You will need to transform between the two systems.
//       Keep in mind that this transformation requires both rotation AND 
//       translation (but no scaling).
//   The following is a good resource for the theory:
//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
//   and the following is a good resource for the actual equation to implement 
//   (look at equation 3.33 http://planning.cs.uiuc.edu/node99.html
void ParticleFilter::updateWeights(
    double sensor_range, 
    double std_landmark[], 
    const std::vector<LandmarkObs> &observations, 
    const Map &map_landmarks) {
    
    double x_std = std_landmark[0];
    double y_std = std_landmark[1];
    for (auto pa : this->particles) {
        vector<LandmarkObs> observations_world;

        for (auto landmark :observations) {
            // transformed from local coordinate to world coordinate
            LandmarkObs obs_t = {
                -1,
                pa.x * cos(pa.theta) - pa.y * sin(pa.theta) + landmark.x,
                pa.x * sin(pa.theta) - pa.y * cos(pa.theta) + landmark.y,
            };
            
            vector<double> distances;
            auto [id1, x1, y1] = obs_t;
            for (auto landmark : map_landmarks.landmark_list) {
                auto [id2, x2, y2] = landmark;
                double distance = this->distance(x1, y1, x2, y2);
                if (distance <= sensor_range)
                    distances.push_back(distance);
            }
            vector<double>::iterator result = min_element(
                begin(distances), end(distances));
            auto lm = map_landmarks
                .landmark_list[::distance(begin(distances), result)];
            obs_t.id = lm.id_i;
            // printf ("ID: %d\n", lm.id_i);
            observations_world.push_back(obs_t);
        }
        double prod = 1.0;
        for (auto landmark : observations_world) {
            printf ("ID: %d\n", pa.id);
            auto map_landmark = map_landmarks.landmark_list[landmark.id];
            prod *= this->gaussian_probability(pa.x, x_std, map_landmark.x_f);
            prod *= this->gaussian_probability(pa.y, y_std, map_landmark.y_f);
            printf ("X: %4.2f  %4.2f  %4.2f  %4.2f\n", pa.x, x_std, map_landmark.x_f,
                    gaussian_probability(pa.x, x_std, map_landmark.x_f));
            printf ("Y: %4.2f  %4.2f  %4.2f  %4.2f\n", pa.y, y_std, map_landmark.y_f,
                    gaussian_probability(pa.y, y_std, map_landmark.y_f));
        }
        
        pa.weight = prod;
    }
}

void ParticleFilter::resample() {
    std::random_device rd;
    std::mt19937 gen(rd());
    vector<double> weights;
    for (auto particle : this->particles)
        weights.push_back(particle.weight);
    std::discrete_distribution<> d(weights.begin(), weights.end());
    vector<Particle> particles_new;
    for(int n=0; n<num_particles; ++n) {
        particles_new.push_back(this->particles[d(gen)]);
    }
    this->particles = particles_new;
    this->num_particles = particles_new.size();
}

Particle ParticleFilter::SetAssociations(
    Particle particle, 
    std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    // particle: the particle to assign each listed association, 
    // and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
     particle.sense_x = sense_x;
     particle.sense_y = sense_y;

     return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
