#ifndef MVEC_H
#define MVEC_H

typedef struct mvec{
	int x;
	int y;
	unsigned int diff;
	mvec operator= (const mvec &m){
		x = m.x;
		y = m.y;
		diff = m.diff;
		return *this;
	}
	mvec operator+= (const mvec &m){
		if(m.diff == diff){
			if((m.x*m.x+m.y*m.y)<(x*x+y*y)){
				x = m.x;
				y = m.y;
				diff = m.diff;
			}
		}
		if(m.diff < diff){
			x = m.x;
			y = m.y;
			diff = m.diff;
		}
		return *this;
	}
} mvec;

#endif // MVEC_H
