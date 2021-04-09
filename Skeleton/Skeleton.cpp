//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char *const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char *const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU
const float EPSILON = 0.01f;

struct Material {
	vec3 ka, kd, ks;
	float shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd *M_PI), kd(_kd), ks(_ks) { shininess = _shininess; };
};

struct Hit {
	float t;
	vec3 position, normal;
	Material *material;
	Hit() { t = -1; }

};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) {
		start = _start; dir = normalize(_dir);
	}
};

class Intersectable {
protected:
	Material *material;
public:
	virtual Hit intersect(const Ray &ray) = 0;

};

struct Dodecahedron : public Intersectable {
	std::vector<vec3> vertices = {
		{0.0f, 0.618f, 1.618f},
		{0.0f, -0.618, 1.618f},
		{0.0f, -0.618f, -1.618f},
		{0.0f, 0.618f, -1.618f},
		{1.618f, 0.0f, 0.618f},
		{-1.618f, 0.0f, 0.618f},
		{-1.618f, 0.0f, -0.618f},
		{1.618f, 0.0f, -0.618f},
		{0.618f, 1.618f, 0.0f},
		{-0.618f, 1.618f, 0.0f},
		{-0.618f, -1.618f, 0.0f},
		{0.618f, -1.618f, 0.0f},
		{1.0f, 1.0f, 1.0f},
		{-1.0f, 1.0f, 1.0f},
		{-1.0f, -1.0f, 1.0f},
		{1.0f, -1.0f, 1.0f},
		{1.0f, -1.0f, -1.0f},
		{1.0f, 1.0f, -1.0f},
		{-1.0f, 1.0f, -1.0f},
		{-1.0f, -1.0f, -1.0f}
	};

	std::vector<std::vector<int>> faces = {
			{1,	 2,  16, 5,  13},
			{1,  13, 9,  10, 14},
			{1,  14, 6,  15, 2},
			{2,  15, 11, 12, 16},
			{3,  4,  18, 8,  17},
			{3,  17, 12, 11, 20},
			{3,  20, 7,  19, 4},
			{19, 10, 9,  18, 4},
			{16, 12, 17, 8,  5},
			{5,  8,  18, 9,  13},
			{14, 10, 19, 7,  6},
			{6,  7,  20, 11, 15}
	};

	Dodecahedron() {
		//TODO: ha nagyon fucked a tükrözõdés akkor ezeket kell baszni
		vec3 kd(1.5f, 1.5f, 1.5f), ks(50, 50, 50);
		material = new Material(kd, ks, 50);
	}

	std::pair<vec3, vec3> getObjectPlane(int faceIndex) {
		vec3 p1 = vertices[faces[faceIndex][0] - 1];
		vec3 p2 = vertices[faces[faceIndex][1] - 1];
		vec3 p3 = vertices[faces[faceIndex][2] - 1];
		
		vec3 normal = cross(p2 - p1, p3 - p1);
		if (dot(p1, normal) < 0)	//ha kifelé mutat meginvertáljuk
			normal = -normal;
		//TODO: szkél??
		return std::pair<vec3, vec3>(p1, normal);
	}

	Hit intersect(const Ray &ray, Hit hit) {
		for (int i = 0; i < faces.size(); i++) {
			std::pair<vec3, vec3> planePair = getObjectPlane(i);
			vec3 normal = planePair.first;	//the normal of the plane
			vec3 p1 = planePair.second;	//first point of face

			float ti;	//distance on ray from start 'till first intersection
			if (abs(dot(normal, ray.dir)) > EPSILON) {
				ti = dot(p1 - ray.start, normal) / dot(normal, ray.dir);
			} else {
				ti = -1.0f;
			}

			if (ti <= EPSILON || (ti > hit.t && hit.t > 0)) continue;	//is current ti closer than hit.t?
			vec3 pintersect = ray.start + ray.dir * ti;

			//now we check if intersection is outside of the face or not
			bool outside = false;
			for (int j = 0; j < faces.size(); j++) {
				if (i == j) continue;
				std::pair<vec3, vec3>otherPlanePair = getObjectPlane(j);
				vec3 otherNormal = otherPlanePair.first;
				vec3 otherPoint = otherPlanePair.second;
				if (dot(normal, pintersect - otherPoint) > 0) {	//checks if normalvector is pointing inwards
					outside = true;
					break;
				}
			}
			if (!outside) {
				hit.t = ti;
				hit.position = pintersect;
				hit.normal = normalize(normal);
				hit.material = material;
			}
		}
		return hit;
		
	}
};


//TODO: megérteni mi történik itt
struct Camera
{
    vec3 eye, lookat, right, pvup, rvup;
    float fov = 45 * (float)M_PI / 180;
 
    Camera() : eye(0, 1, 1), pvup(0, 0, 1), lookat(0, 0, 0) { set(); }
    void set()
    {
        vec3 w = eye - lookat;
        float f = length(w);
        right = normalize(cross(pvup, w)) * f * tanf(fov / 2);
        rvup = normalize(cross(w, right)) * f * tanf(fov / 2);
    }
    void Animate(float t)
    {
		//rotates the camera around lookat ( = (0, 0, 0))
        float r = sqrtf(eye.x * eye.x + eye.y * eye.y);
        eye = vec3(r * cos(t) + lookat.x, r * sin(t) + lookat.y, eye.z);
        set();
    }
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) : direction(_direction), Le(_Le) {};
};

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	Dodecahedron dodecahedron();
	//TODO: create scene

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	glBindVertexArray(vao);  // Draw call
	glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, 3 /*# Elements*/);

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char *buttonStat;
	switch (state) {
		case GLUT_DOWN: buttonStat = "pressed"; break;
		case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
		case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
		case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
		case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
