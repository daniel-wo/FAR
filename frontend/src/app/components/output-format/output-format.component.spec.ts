import { ComponentFixture, TestBed } from '@angular/core/testing';

import { OutputFormatComponent } from './output-format.component';

describe('OutputFormatComponent', () => {
  let component: OutputFormatComponent;
  let fixture: ComponentFixture<OutputFormatComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ OutputFormatComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(OutputFormatComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
