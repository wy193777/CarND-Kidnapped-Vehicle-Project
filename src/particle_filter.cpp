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

double ParticleFilter::gaussian_weight(
    double p_x,
    double p_y,
    double mu_x,
    double mu_y,
    double std_x,
    double std_y) {
    
    double c = 1.0 / (2.0 * M_PI * std_x * std_y);
    double x_pow = pow(p_x - mu_x, 2) / (2 * pow(std_x, 2));
    double y_pow = pow(p_y - mu_y, 2) / (2 * pow(std_y, 2));
    return c * exp( - x_pow - y_pow);
}


double ParticleFilter::distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0));
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 10;
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
    
    double curve = velocity / yaw_rate;
    
    for (auto &pa : this->particles) {
        double new_x, new_y, new_theta;
        if (abs(yaw_rate) >= 0.0001) {
            new_x = pa.x + curve * (sin(pa.theta + yaw_rate * delta_t) - sin(pa.theta));
            new_y = pa.y + curve * (cos(pa.theta) - cos(pa.theta + yaw_rate * delta_t));
            new_theta = pa.theta + yaw_rate * delta_t;
        } else {
            new_x = pa.x + velocity * delta_t * cos(pa.theta);
            new_y = pa.y + velocity * delta_t * sin(pa.theta);
            new_theta = pa.theta;
        }
        pa.x = gaussian_random(new_x, std_pos[0]);
        pa.y = gaussian_random(new_y, std_pos[1]);
        pa.theta = gaussian_random(new_theta, std_pos[2]);
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
    for (int idx = 0; idx < this->particles.size(); idx++) {
        Particle pa = particles[idx];
        vector<double> sense_x;
        vector<double> sense_y;
        vector<int> associations;
        pa.weight = 1.0;
        for (auto &ob :observations) {
            // transformed from local coordinate to world coordinate
            
            double world_x = ob.x * cos(pa.theta) - ob.y * sin(pa.theta) + pa.x;
            double world_y = ob.x * sin(pa.theta) + ob.y * cos(pa.theta) + pa.y;
            double lm_to_measure = numeric_limits<double>::max();
            int cloest_lm_id = 0;
            
            Map::single_landmark_s closest_lm;
            for (auto &landmark : map_landmarks.landmark_list) {
                if (this->distance(pa.x, pa.y, landmark.x_f, landmark.y_f) > sensor_range) {
                    continue;
                }
                
                double dist = this->distance(world_x, world_y, landmark.x_f, landmark.y_f);
                if (dist < lm_to_measure) {
                    lm_to_measure = dist;
                    cloest_lm_id = landmark.id_i;
                    closest_lm = landmark;
                }
            }
            sense_x.push_back(world_x);
            sense_y.push_back(world_y);
            associations.push_back(cloest_lm_id);
        
            long double multiplier = this->gaussian_weight(
                world_x, world_y, 
                closest_lm.x_f, closest_lm.y_f, x_std, y_std);
            pa.weight *= multiplier;
        }
        
        pa = SetAssociations(pa, associations, sense_x, sense_y);
        weights.push_back(pa.weight);
        particles[idx] = pa;
    }
}

void ParticleFilter::resample() {
    std::discrete_distribution<> d(this->weights.begin(), this->weights.end());
    vector<Particle> particles_new;
    for(int n=0; n<num_particles; ++n) {
        particles_new.push_back(this->particles[d(ENGINE)]);
    }
    this->particles = particles_new;
    weights.clear();
}

Particle ParticleFilter::SetAssociations(
    Particle particle, 
    std::vector<int> associations, 
    std::vector<double> sense_x, 
    std::vector<double> sense_y) {
    // particle: the particle to assign each listed association, 
    // and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
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
